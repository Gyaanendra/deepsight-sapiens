---
title: KnightSight EdgeVision ANPR
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# 🚗 KnightSight EdgeVision: ANPR Pipeline

Automatic Number Plate Recognition system built for the KnightSight EdgeVision Challenge.

**Pipeline:** Detect Vehicle → Crop ROI → Detect License Plate → OCR

## 📁 Directory Structure
- `app.py` — Streamlit inference dashboard
- `models/` — YOLO weights (best.pt, best.onnx, yolo11n.pt)
- `configs/` — Dataset and training configuration
- `runs/` — Training outputs, metrics, and weights
- `scripts/` — Export and utility scripts
- `notebooks/` — Training and augmentation research
- `explain.md` — Full technical report

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Key Components
- **Vehicle Detection**: YOLOv11n (COCO pre-trained)
- **Plate Detection**: Custom fine-tuned YOLOv11n (mAP@50: 99.49%)
- **OCR Engine**: Fast-Plate-OCR / EasyOCR
- **Night Vision**: CLAHE enhancement

## 📊 Model Performance
| Metric | Value |
|---|---|
| mAP@50 | 99.49% |
| mAP@50-95 | 72.91% |
| Precision | 99.90% |
| Recall | 99.45% |
| F1 Score | 99.67% |

---
*Developed for KnightSight EdgeVision Challenge · Ultralytics + Streamlit*
