import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import easyocr
import time
import tempfile
import os

st.set_page_config(
    page_title="KnightSight | ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #080c14; }

/* Inference timer banner */
.timer-banner {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border: 1px solid rgba(0,201,255,0.3);
    border-radius: 12px;
    padding: 14px 24px;
    display: flex; align-items: center; gap: 20px;
    margin-bottom: 16px;
}
.timer-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg,#00c9ff,#92fe9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.timer-label { color: #7a9db5; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; }

/* Crop cards */
.crop-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px; margin-bottom: 10px;
}
.ocr-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem; font-weight: 700; letter-spacing: 6px;
    background: linear-gradient(90deg,#00c9ff,#92fe9d);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 6px;
}
.pill-vehicle { background: rgba(255,140,0,0.2); color: #ff8c00; border: 1px solid #ff8c00; }
.pill-fallback { background: rgba(0,180,255,0.2); color: #00b4ff; border: 1px solid #00b4ff; }

/* Section headers */
.sec-header { font-size: 0.7rem; color:#4a6fa5; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
PT_PATH   = "models/best.pt"
ONNX_PATH = "models/best.onnx"
VMODEL    = "models/yolo11n.pt"
VCL       = [2, 3, 5, 7]   # car, motorcycle, bus, truck

# ── Loaders (cached) ─────────────────────────────────────────────────────────
@st.cache_resource
def load_plate(path): return YOLO(path, task="detect")

@st.cache_resource
def load_vehicle(): return YOLO(VMODEL)

@st.cache_resource
def load_fast_ocr(): return LicensePlateRecognizer("cct-s-v2-global-model")

@st.cache_resource
def load_fast_ocr(): return LicensePlateRecognizer("cct-s-v2-global-model")

# ── OCR dispatch ─────────────────────────────────────────────────────────────
def do_ocr(crop_bgr, fast_ocr):
    try:
        res = fast_ocr.run(crop_bgr)
        txt = res[0].plate if (isinstance(res, list) and res) else str(res)
        return txt.upper().replace(".", "").replace("-", "").replace("_", "").strip()
    except Exception:
        return ""

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def pad_crop(img, px1, py1, px2, py2, pad=15):
    h, w = img.shape[:2]
    return img[max(0,py1-pad):min(h,py2+pad), max(0,px1-pad):min(w,px2+pad)]

def upscale(crop, scale=4):
    """Lanczos upscale + denoising — improves OCR on small plate crops."""
    h, w = crop.shape[:2]
    if h == 0 or w == 0: return crop
    # Apply bilateral filter to denoise while keeping edges sharp
    denoised = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)
    return cv2.resize(denoised, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

# ── Core pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(img_bgr, plate_m, vehicle_m, conf, use_night, fast_ocr):
    if use_night:
        img_bgr = apply_clahe(img_bgr)
    annotated = img_bgr.copy()
    detections = []
    t0 = time.time()

    # 1. Vehicle detection
    v_res = vehicle_m.predict(img_bgr, conf=0.3, classes=VCL, verbose=False)
    v_boxes = [b.xyxy[0].cpu().numpy().astype(int) for b in v_res[0].boxes]

    plate_found = False
    for vb in v_boxes:
        vx1, vy1, vx2, vy2 = vb
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 140, 0), 2)
        cv2.putText(annotated, "Vehicle", (vx1, vy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 140, 0), 2)
        vcrop = img_bgr[vy1:vy2, vx1:vx2]
        if vcrop.size == 0: continue

        # 2. Plate in vehicle crop
        p_res = plate_m.predict(vcrop, conf=conf, verbose=False)
        for pb in p_res[0].boxes:
            plate_found = True
            px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy().astype(int)
            pconf = float(pb.conf[0])
            plate_crop = pad_crop(vcrop, px1, py1, px2, py2)
            if plate_crop.size == 0: continue
            plate_crop_up = upscale(plate_crop, scale=4)

            # 3. OCR on upscaled crop
            ocr_t = do_ocr(plate_crop_up, fast_ocr)
            detections.append({
                "conf": pconf, "ocr": ocr_t,
                "vehicle_crop": vcrop.copy(),
                "plate_crop": plate_crop.copy(),
                "source": "vehicle"
            })
            ax1, ay1 = vx1+px1, vy1+py1
            ax2, ay2 = vx1+px2, vy1+py2
            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 255, 80), 3)
            cv2.putText(annotated, f"{ocr_t} {pconf:.2f}",
                        (ax1, ay1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)

    # 4. Fallback — direct plate detect on full image
    if not plate_found:
        p_res = plate_m.predict(img_bgr, conf=conf, verbose=False)
        for pb in p_res[0].boxes:
            px1, py1, px2, py2 = pb.xyxy[0].cpu().numpy().astype(int)
            pconf = float(pb.conf[0])
            plate_crop = pad_crop(img_bgr, px1, py1, px2, py2)
            if plate_crop.size == 0: continue
            plate_crop_up = upscale(plate_crop, scale=4)
            ocr_t = do_ocr(plate_crop_up, fast_ocr)
            detections.append({
                "conf": pconf, "ocr": ocr_t,
                "vehicle_crop": None,
                "plate_crop": plate_crop.copy(),
                "source": "fallback"
            })
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 180, 255), 3)
            cv2.putText(annotated, f"[FB] {ocr_t} {pconf:.2f}",
                        (px1, py1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)

    inf_ms = (time.time() - t0) * 1000
    return annotated, detections, inf_ms, len(v_boxes)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ KnightSight Controls")
    st.markdown("---")
    engine = st.selectbox("🔧 Plate Model Engine", ["ONNX (Optimized)", "PyTorch (Original)"])
    ocr_engine_name = "Fast-Plate-OCR" # Hardcoded since it's the only one left
    conf_thr = st.slider("🎯 Plate Confidence", 0.10, 1.0, 0.40, 0.05)
    use_night = st.checkbox("🌙 Night Vision (CLAHE)", value=False)
    st.markdown("---")
    st.markdown("""
**Pipeline**
1. 🚙 Detect vehicle (YOLO COCO)
2. ✂️ Crop vehicle ROI
3. 🔍 Detect plate in crop
4. 🔤 OCR on plate
5. 🔄 Fallback → full-image
""")

# ── Load models once (lazy EasyOCR to save RAM on HF Spaces) ─────────────────
plate_model   = load_plate(ONNX_PATH if "ONNX" in engine else PT_PATH)
vehicle_model = load_vehicle()
fast_ocr      = load_fast_ocr()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚗 KnightSight EdgeVision ANPR")
st.markdown("<p style='color:#4a6fa5;margin-top:-12px'>Automatic Number Plate Recognition · Vehicle → Plate → OCR pipeline</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
img_tab, vid_tab = st.tabs(["📷  Image Inference", "🎬  Video Inference"])

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE TAB
# ═══════════════════════════════════════════════════════════════════════════════
with img_tab:
    uploaded = st.file_uploader("Upload vehicle image", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if uploaded:
        img_pil  = Image.open(uploaded)
        img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Running ANPR pipeline…"):
            annotated, dets, inf_ms, n_vehicles = run_pipeline(
                img_bgr, plate_model, vehicle_model,
                conf_thr, use_night, fast_ocr
            )

        # ── Inference time banner ─────────────────────────────────────────────
        fallback_used = any(d["source"] == "fallback" for d in dets)
        st.markdown(f"""
<div class="timer-banner">
  <div>
    <div class="timer-label">Inference Time</div>
    <div class="timer-val">{inf_ms:.0f} ms</div>
  </div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:48px;margin:0 8px"></div>
  <div>
    <div class="timer-label">Vehicles Detected</div>
    <div class="timer-val" style="font-size:1.6rem">{n_vehicles}</div>
  </div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:48px;margin:0 8px"></div>
  <div>
    <div class="timer-label">Plates Found</div>
    <div class="timer-val" style="font-size:1.6rem">{len(dets)}</div>
  </div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:48px;margin:0 8px"></div>
  <div>
    <div class="timer-label">Mode</div>
    <div class="timer-val" style="font-size:1rem;margin-top:6px">{'🔄 Fallback' if fallback_used else '🚙 Vehicle'}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Layout: annotated + results ───────────────────────────────────────
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("<div class='sec-header'>Annotated Frame</div>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')

        with col2:
            st.markdown("<div class='sec-header'>Detections</div>", unsafe_allow_html=True)
            if not dets:
                st.warning("No plates detected.")
            for i, d in enumerate(dets):
                pill_cls = "pill-fallback" if d["source"] == "fallback" else "pill-vehicle"
                pill_lbl = "🔄 Fallback" if d["source"] == "fallback" else "🚙 Via Vehicle"
                st.markdown(f"""
<div class="crop-card">
  <span class="pill {pill_cls}">{pill_lbl}</span>
  <b>Plate #{i+1}</b>
""", unsafe_allow_html=True)

                # Show vehicle crop if available
                if d["vehicle_crop"] is not None:
                    st.markdown("<div class='sec-header'>Vehicle Crop</div>", unsafe_allow_html=True)
                    st.image(cv2.cvtColor(d["vehicle_crop"], cv2.COLOR_BGR2RGB),
                             width='stretch')

                st.markdown("<div class='sec-header'>Plate Crop</div>", unsafe_allow_html=True)
                st.image(cv2.cvtColor(d["plate_crop"], cv2.COLOR_BGR2RGB),
                         width='stretch')

                st.markdown(f"<div class='ocr-text'>{d['ocr'] or '???'}</div>", unsafe_allow_html=True)
                st.caption(f"Confidence: {d['conf']:.2f}  ·  OCR: {ocr_engine_name}")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("")

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#3a5570">
          <div style="font-size:4rem">📷</div>
          <div style="font-size:1.2rem;margin-top:12px">Upload an image to begin inference</div>
          <div style="font-size:0.85rem;margin-top:6px;color:#2a4560">
            Vehicle detect → Crop → Plate detect → OCR
          </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO TAB
# ═══════════════════════════════════════════════════════════════════════════════
with vid_tab:
    vid_file = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"],
                                label_visibility="collapsed")
    sample_every = st.slider("Process every N frames", 1, 30, 5,
                             help="Higher = faster but less coverage")

    if vid_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25

        st.info(f"Video: {total_frames} frames · {fps:.1f} fps · processing every {sample_every}th frame")

        run_btn = st.button("▶️ Run Video Inference", type="primary", use_container_width=True)

        if run_btn:
            frame_display = st.empty()
            metrics_row   = st.empty()
            plates_area   = st.expander("📋 All detected plates", expanded=True)

            all_plates = []
            frame_idx  = 0
            prog       = st.progress(0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                prog.progress(min(frame_idx / total_frames, 1.0))

                if frame_idx % sample_every != 0:
                    continue

                annotated, dets, inf_ms, n_v = run_pipeline(
                    frame, plate_model, vehicle_model,
                    conf_thr, use_night, fast_ocr
                )

                frame_display.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {frame_idx}/{total_frames} · {inf_ms:.0f}ms",
                                    width='stretch')

                metrics_row.markdown(f"""
<div class="timer-banner">
  <div><div class="timer-label">Frame</div><div class="timer-val" style="font-size:1.4rem">{frame_idx}/{total_frames}</div></div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:40px;margin:0 8px"></div>
  <div><div class="timer-label">Latency</div><div class="timer-val" style="font-size:1.4rem">{inf_ms:.0f}ms</div></div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:40px;margin:0 8px"></div>
  <div><div class="timer-label">Vehicles</div><div class="timer-val" style="font-size:1.4rem">{n_v}</div></div>
  <div style="border-left:1px solid rgba(255,255,255,0.1);height:40px;margin:0 8px"></div>
  <div><div class="timer-label">Plates (frame)</div><div class="timer-val" style="font-size:1.4rem">{len(dets)}</div></div>
</div>
""", unsafe_allow_html=True)

                for d in dets:
                    all_plates.append({"frame": frame_idx, **d})

            cap.release()
            os.unlink(tmp_path)
            prog.empty()

            # Summary
            with plates_area:
                st.markdown(f"**Total plates detected across all frames: {len(all_plates)}**")
                for i, d in enumerate(all_plates):
                    cols = st.columns([1, 2, 3])
                    cols[0].caption(f"Frame {d['frame']}")
                    cols[1].image(cv2.cvtColor(d["plate_crop"], cv2.COLOR_BGR2RGB))
                    cols[2].markdown(f"<div class='ocr-text' style='font-size:1.2rem'>{d['ocr'] or '???'}</div>",
                                     unsafe_allow_html=True)
                    cols[2].caption(f"Conf {d['conf']:.2f} · {d['source']}")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#3a5570">
          <div style="font-size:4rem">🎬</div>
          <div style="font-size:1.2rem;margin-top:12px">Upload a video to run inference</div>
          <div style="font-size:0.85rem;margin-top:6px;color:#2a4560">
            Processes sampled frames · shows live annotated feed
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#2a4560;font-size:0.78rem;margin-top:32px;padding:16px;
border-top:1px solid rgba(255,255,255,0.05)'>
KnightSight EdgeVision ANPR · Ultralytics YOLO + Fast-Plate-OCR / EasyOCR · Streamlit
</div>""", unsafe_allow_html=True)
