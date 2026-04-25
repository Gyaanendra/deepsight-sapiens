import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer
import time
import os

# Set page configuration
st.set_page_config(
    page_title="KnightSight | EdgeVision ANPR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stHeader {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background: linear-gradient(45deg, #00C9FF, #92FE9D);
        color: black;
        font-weight: bold;
        border: none;
    }
    .ocr-result {
        font-size: 24px;
        font-weight: bold;
        color: #92FE9D;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Constants & Paths ---
PT_MODEL_PATH = "models/best.pt"
ONNX_MODEL_PATH = "models/best.onnx"
VEHICLE_MODEL_PATH = "models/yolo11n.pt"

# --- Cached Loaders ---
@st.cache_resource
def load_models(model_path):
    try:
        # Load custom plate detector (works for both .pt and .onnx)
        plate_model = YOLO(model_path, task='detect')
        # Load standard vehicle detector
        vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        return plate_model, vehicle_model
    except Exception as e:
        st.error(f"Error loading models from {model_path}: {e}")
        return None, None


@st.cache_resource
def load_ocr():
    # specialized plate OCR model
    return LicensePlateRecognizer('cct-s-v2-global-model')

# --- Helper Functions ---
def apply_night_vision(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def letterbox_image(image, expected_size):
    """Resize image with padding to maintain aspect ratio."""
    ih, iw = image.shape[:2]
    ew, eh = expected_size, expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # Create a new gray background image
    new_img = np.full((eh, ew, 3), 114, dtype=np.uint8)
    # Paste the original image into the center
    dy = (eh - nh) // 2
    dx = (ew - nw) // 2
    new_img[dy:dy+nh, dx:dx+nw, :] = image
    return new_img, (dy, dx, scale)

def run_pipeline(image, plate_model, vehicle_model, ocr_engine, conf_threshold, do_vehicle_det, use_night_mode):
    results_data = []
    
    # 1. Convert and Letterbox
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_ready, (dy, dx, scale) = letterbox_image(img_bgr, 640)
    
    if use_night_mode:
        img_ready = apply_night_vision(img_ready)
        
    annotated_img = img_ready.copy()
    start_time = time.time()
    
    # 2. Vehicle Detection
    if do_vehicle_det:
        v_results = vehicle_model.predict(img_ready, conf=0.25, classes=[2, 3, 5, 7], verbose=False)
        for v in v_results[0].boxes:
            v_box = v.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_img, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (255, 100, 0), 2)
    
    # 3. Plate Detection
    p_results = plate_model.predict(img_ready, conf=conf_threshold, verbose=False)
    inference_time = (time.time() - start_time) * 1000 
    
    padding_pct = 0.05 # fast-plate-ocr likes cleaner but slightly padded crops
    
    for p in p_results[0].boxes:
        p_box = p.xyxy[0].cpu().numpy().astype(int)
        p_conf = p.conf[0].item()
        
        # Crop from the same 'img_ready' used for detection (640x640)
        h_ready, w_ready = img_ready.shape[:2]
        pad_x = int((p_box[2] - p_box[0]) * padding_pct)
        pad_y = int((p_box[3] - p_box[1]) * padding_pct)
        
        x1 = max(0, p_box[0] - pad_x)
        y1 = max(0, p_box[1] - pad_y)
        x2 = min(w_ready, p_box[2] + pad_x)
        y2 = min(h_ready, p_box[3] + pad_y)
        
        crop = img_ready[y1:y2, x1:x2]
        
        # Safety check: skip if crop is empty
        if crop is None or crop.size == 0:
            continue
            
        ocr_text = ""
        try:
            # fast-plate-ocr returns a list of PlatePrediction objects
            ocr_results = ocr_engine.run(crop)
            
            if isinstance(ocr_results, list) and len(ocr_results) > 0:
                # Access the .plate attribute of the first prediction
                ocr_text = ocr_results[0].plate
            else:
                ocr_text = str(ocr_results)
            
            # Clean: Uppercase, remove noise
            ocr_text = str(ocr_text).upper().replace(".", "").replace("-", "").replace("_", "").strip()
        except Exception:
            ocr_text = ""

        results_data.append({
            "Box": [x1, y1, x2, y2],
            "Conf": p_conf,
            "OCR": ocr_text,
            "Crop": crop
        })
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"PLATE {p_conf:.2f} | {ocr_text}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated_img, results_data, inference_time

# --- Sidebar ---
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("### 🔥 Model Settings")
model_format = st.sidebar.selectbox("📦 Engine", ["ONNX (Optimized)", "PyTorch (Original)"])
conf_threshold = st.sidebar.slider("🎯 Confidence", 0.1, 1.0, 0.45)

st.sidebar.markdown("### 🛠️ Enhancements")
do_vehicle_det = st.sidebar.checkbox("🚚 Vehicle Detection", value=True)
use_night_mode = st.sidebar.checkbox("🌙 Night Vision (CLAHE)", value=False)

st.sidebar.markdown("---")
st.sidebar.info("""
**KnightSight EdgeVision**
- Engine: ONNX / PyTorch
- OCR: Fast-Plate-OCR (SOTA)
- Bonus: CLAHE Night Vision
""")

# --- Main Page ---
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Determine model path based on selection
    selected_path = ONNX_MODEL_PATH if "ONNX" in model_format else PT_MODEL_PATH
    
    # Load Models
    plate_model, vehicle_model = load_models(selected_path)
    reader = load_ocr()
    
    if plate_model and reader:
        with st.spinner(f"Processing with {model_format}..."):
            annotated_img, detections, inf_time = run_pipeline(
                image, plate_model, vehicle_model, reader, conf_threshold, do_vehicle_det, use_night_mode
            )
        
        # Layout: 2 Columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🖼️ Detection Performance")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Metrics Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Latency", f"{inf_time:.1f} ms")
            with m_col2:
                st.metric("Detections", len(detections))
            with m_col3:
                st.metric("Precision", "High")
        
        with col2:
            st.markdown("### 📝 Structured Results")
            if not detections:
                st.warning("No plates detected in frame.")
            else:
                for i, det in enumerate(detections):
                    st.markdown(f"**Object #{i+1}**")
                    st.image(cv2.cvtColor(det['Crop'], cv2.COLOR_BGR2RGB), width=200)
                    st.markdown(f"<p class='ocr-result'>{det['OCR'] if det['OCR'] else '---'}</p>", unsafe_allow_html=True)
                    st.json({
                        "Confidence": f"{det['Conf']:.2f}",
                        "OCR_Text": det['OCR']
                    })
                    st.divider()

else:
    st.markdown("""
    ### 👋 Welcome to the EdgeVision Pipeline
    Please upload an image from the sidebar to begin.
    
    **System Capabilities:**
    1. **Vehicle Localization**: Identifies cars, bikes, and trucks in the scene.
    2. **Plate Detection**: High-precision localization of Indian standard plates.
    3. **OCR Recognition**: Extracts alphanumerics under various lighting conditions.
    4. **Optimized for Edge**: Runs on YOLOv11 nano architecture.
    """)
    
    # Show sample if possible (optional)
    # st.image("docs/visualizations/training_metrics.png", caption="Training Metrics", width=600)

footer_html = """<div style='text-align: center;'>
<p>Developed for KnightSight Challenge | Powered by Ultralytics & Streamlit</p>
</div>"""
st.markdown(footer_html, unsafe_allow_html=True)
