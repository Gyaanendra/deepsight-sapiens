# 🚗 KnightSight EdgeVision: ANPR Pipeline

Professional Vehicle Intelligence and License Plate Recognition system optimized for Edge deployment.

## 📁 Directory Structure
- `app.py`: Main Streamlit Inference Dashboard.
- `configs/`: Dataset and model configuration files (YAML).
- `data/`: Raw and processed datasets.
- `docs/`: PDFs, project documentation, and backups.
- `models/`: Downloaded base models (e.g., YOLOv11n).
- `notebooks/`: Training and data augmentation research.
- `runs/`: Training outputs, logs, and fine-tuned weights.
- `scripts/`: Utility scripts (Export, Inspections).
- `tests/`: Module testing and validation scripts.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Inference Dashboard
```bash
streamlit run app.py
```

### 3. Edge Optimization (Export to ONNX)
```bash
python scripts/export_onnx.py
```

## 🛠️ Key Components
- **Vehicle Detection**: YOLOv11 nano (COCO filtered).
- **Plate Detection**: Custom-tuned YOLOv11 nano.
- **OCR Engine**: EasyOCR with Grayscale + CLAHE Pre-processing.
- **Robustness**: Integrated Night Vision enhancement.

---
*Developed for KnightSight EdgeVision Challenge.*
