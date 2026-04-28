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

<div align="center">

# 🚗 KnightSight EdgeVision — ANPR Pipeline

**Automatic Number Plate Recognition · Edge-Optimized · Real-Time Inference**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-blue?logo=udacity&logoColor=white)](https://docs.ultralytics.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Space-yellow)](https://huggingface.co/spaces/Gyaanendra/deepsight-sapiens)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Pipeline Architecture](#-pipeline-architecture)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Training](#-training)
- [ONNX Export](#-onnx-export)
- [Creating Submissions](#-creating-submissions)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Technical Deep Dive](#-technical-deep-dive)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

**KnightSight EdgeVision** is a high-accuracy, edge-optimized Automatic Number Plate Recognition (ANPR) system built for the **KnightSight EdgeVision Challenge**. It implements a two-stage detection pipeline — first detecting vehicles in a scene, then localizing license plates within each vehicle ROI, and finally performing OCR to extract the plate text.

The system achieves **99.49% mAP@50** with a lightweight 2.6M parameter model, making it suitable for real-time edge deployment while maintaining near-perfect detection accuracy.

---

## 🌐 Live Demo

> **Try it live on Hugging Face Spaces:**  
> 🔗 [https://huggingface.co/spaces/Gyaanendra/deepsight-sapiens](https://huggingface.co/spaces/Gyaanendra/deepsight-sapiens)

Upload any vehicle image or video and get instant plate detection + OCR results.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🚙 **Two-Stage Detection** | Vehicle detection (COCO YOLOv11n) → Plate detection (custom fine-tuned YOLOv11n) |
| 🔤 **Dual OCR Engines** | Fast-Plate-OCR (~5ms) + GPT-4o-mini via OpenRouter as a fallback |
| 🌙 **Night Vision** | CLAHE contrast enhancement in LAB colorspace for low-light scenarios |
| ⚡ **ONNX Optimized** | Export to ONNX for cross-platform edge inference (~6ms per frame) |
| 🎬 **Video Inference** | Frame-by-frame video processing with configurable sample rate |
| 🔥 **GPU Acceleration** | Automatic CUDA detection with seamless CPU fallback |
| 🔄 **Smart Fallback** | If no vehicle is detected, runs plate detection on the full frame |
| 📊 **Real-Time Metrics** | Live inference time, vehicle count, and plate count displayed in-dashboard |

---

## 🏗️ Pipeline Architecture

```
Input Image / Video Frame
        │
        ▼
┌─────────────────────────┐
│   Vehicle Detector      │  YOLOv11n (COCO pre-trained)
│   conf ≥ 0.30           │  Classes: car, motorcycle, bus, truck
└──────────┬──────────────┘
           │  vehicle crops (ROIs)
           ▼
┌─────────────────────────┐
│   Plate Detector        │  Custom fine-tuned YOLOv11n
│   conf ≥ 0.40           │  (ONNX or PyTorch selectable)
└──────────┬──────────────┘
           │  plate bounding boxes
           ▼
┌─────────────────────────┐
│   Preprocessing         │  Crop + Padding (±15px)
│   + CLAHE (optional)    │  Bilateral filter denoising
│   + 4× Lanczos Upscale  │  Enhances small plate readability
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   OCR Engine            │  Fast-Plate-OCR  (~5ms, default)
│                         │  GPT-4o-mini     (fallback via API)
└──────────┬──────────────┘
           │
           ▼
     Plate Text + Confidence Score

🔄 Fallback: If no vehicles detected → plate detector runs on full frame
🔄 Submission fallback: 90° rotation if no plates found in original orientation
```

**Why two-stage?**
- Reduces false positives (shop signs, road boards that resemble plates)
- Improves plate detector accuracy by providing focused, smaller ROIs
- Reduces effective detection area by ~60–80% → faster inference per frame

---

## 📊 Model Performance

### Detection Metrics (Plate Detector — Epoch 20)

| Metric | Value |
|--------|-------|
| **mAP@50** | **99.49%** |
| **mAP@50-95** | **72.91%** |
| **Precision** | **99.90%** |
| **Recall** | **99.45%** |
| **F1 Score** | **99.67%** |
| Val Box Loss | 1.333 |
| Val Cls Loss | 0.707 |

### Benchmark Comparison

| Model | Params | mAP@50 | mAP@50-95 |
|-------|--------|--------|-----------|
| YOLOv8n COCO general | — | ~37% | ~18% |
| Typical LP detector (academic) | — | 92–96% | 55–65% |
| YOLOv8m fine-tuned LP | 25M | ~97% | ~68% |
| **KnightSight (ours)** | **2.6M** | **99.49%** | **72.91%** |

> Our 2.6M parameter nano model **outperforms** a 25M parameter medium model on both metrics.

### Model Specs

| Property | Value |
|----------|-------|
| Architecture | YOLOv11n (anchor-free, single-class) |
| Parameters | ~2.6M |
| Model Size | 5.3 MB (PT) / 10.5 MB (ONNX) |
| Input Resolution | 480 × 480 px |
| Training | 20 epochs · batch=32 · RTX 4060 |
| Optimizer | AdamW + cosine LR decay |
| Inference Speed | ~8ms (PT) / ~6ms (ONNX) on RTX 4060 |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Detection** | [Ultralytics YOLOv11](https://docs.ultralytics.com) |
| **OCR (Primary)** | [Fast-Plate-OCR](https://github.com/ankandrew/fast-plate-ocr) (`cct-s-v2-global-model`) |
| **OCR (Fallback)** | [GPT-4o-mini](https://openrouter.ai) via OpenRouter API |
| **Framework** | [PyTorch](https://pytorch.org) + [ONNX Runtime](https://onnxruntime.ai) |
| **Image Processing** | [OpenCV](https://opencv.org) (CLAHE, bilateral filter, Lanczos upscale) |
| **Frontend** | [Streamlit](https://streamlit.io) |
| **GPU Support** | CUDA (auto-detected) |
| **Environment** | [python-dotenv](https://pypi.org/project/python-dotenv/) for API key management |

---

## 📁 Project Structure

```
deepsight/
├── app.py                      # 🎯 Streamlit inference dashboard (main entry point)
├── create_submission.py        # 📦 Generate competition submission JSONs
├── visualize_predictions.py    # 📊 Visualize model predictions on test images
│
├── models/                     # 🧠 Model weights
│   ├── best.pt                 #    Fine-tuned plate detector (PyTorch)
│   ├── best.onnx               #    Fine-tuned plate detector (ONNX export)
│   ├── best_lprnet.pth         #    LPRNet weights (experimental)
│   ├── yolo11n.pt              #    Vehicle detector (COCO pre-trained)
│   └── yolo26n.pt              #    YOLOv26n weights (experimental)
│
├── configs/
│   └── knight_sight.yaml       # 📋 Dataset & training configuration
│
├── scripts/
│   ├── export_onnx.py          # 🔄 Export PyTorch → ONNX
│   ├── aug.py                  # 🖼️ Data augmentation utilities
│   ├── plot_results.py         # 📈 Plot training metrics
│   └── inspect_fast_plate_api.py  # 🔍 Fast-Plate-OCR API inspector
│
├── notebooks/
│   ├── train.ipynb             # 🏋️ Model training notebook
│   ├── aug.ipynb               # 🖼️ Augmentation research notebook
│   └── deepsightchallenge-dataset-checkout.ipynb  # 📂 Dataset exploration
│
├── src/                        # 🧩 Source modules
│   ├── License_Plate_Recognition/
│   ├── object_detection/
│   └── semantic_segmentation/
│
├── tests/
│   ├── test_box.py             # ✅ Bounding box unit tests
│   └── test_inference_ocr.py   # ✅ OCR inference tests
│
├── docs/
│   ├── visualizations/
│   │   └── training_metrics.png  # 📈 Training curves visualization
│   └── KnightSight_EdgeVision_Challenge_FINAL.pdf  # 📄 Challenge documentation
│
├── runs/detect/                # 📊 Training outputs, metrics, and weights
├── explain.md                  # 📖 Full technical report (loss functions, architecture, etc.)
│
├── requirements.txt            # 📦 Python dependencies
├── packages.txt                # 📦 System-level apt packages (for HF Spaces)
├── .env                        # 🔐 API keys (not committed)
├── .gitignore                  # 🚫 Git ignore rules
├── .gitattributes              # 📎 Git LFS tracking (*.pt, *.onnx, *.pth, etc.)
├── run.bat                     # 🖥️ Windows launch script
└── run_plot.bat                # 📈 Windows plot metrics script
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.9+
- **CUDA** 11.8+ (optional, for GPU acceleration)
- **Git LFS** (required — model weights are tracked via LFS)

### 1. Clone the Repository

```bash
git lfs install
git clone https://github.com/Gyaanendra/deepsight-sapiens.git
cd deepsight-sapiens
```

### 2. Create a Virtual Environment

```bash
# Using conda
conda create -n knightsight python=3.11 -y
conda activate knightsight

# Or using venv
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **For GPU support**, install the CUDA-compatible PyTorch build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Set Up Environment Variables (Optional)

Create a `.env` file in the project root for the GPT-4o-mini OCR fallback:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> The app works fully without this — Fast-Plate-OCR is the default OCR engine. The OpenRouter key only enables the GPT-4o-mini fallback option in the sidebar.

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Or on Windows, double-click `run.bat`.

---

## 💡 Usage

### Image Inference

1. Open the dashboard at `http://localhost:8501`
2. Select your preferred **Plate Model Engine** (ONNX or PyTorch) in the sidebar
3. Choose an **OCR Engine** (Fast-Plate-OCR or Fallback Model)
4. Adjust the **Plate Confidence** threshold (default: 0.40)
5. Toggle **Night Vision (CLAHE)** for low-light images
6. Upload an image via the **📷 Image Inference** tab
7. View annotated results with detected plates and OCR text

### Video Inference

1. Switch to the **🎬 Video Inference** tab
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Set the **frame sampling rate** (process every N frames)
4. Click **▶️ Run Video Inference**
5. Watch live annotated frames with real-time metrics
6. Review all detected plates in the expandable summary panel

### Sidebar Controls

| Control | Description | Default |
|---------|-------------|---------|
| Plate Model Engine | ONNX (faster) or PyTorch (original) | ONNX |
| OCR Engine | Fast-Plate-OCR or GPT-4o-mini Fallback | Fast-Plate-OCR |
| Plate Confidence | Detection threshold (0.10 – 1.00) | 0.40 |
| Night Vision | CLAHE contrast enhancement | Off |

---

## 🏋️ Training

### Dataset

- **Classes:** 1 (`license_plate`)
- **Split:** Scene-based (no data leakage between train/val)
- **Format:** YOLO `.txt` labels (normalized `x_center y_center width height`)

### Training Configuration

```yaml
model:          yolo11n.pt        # nano — 2.6M params, COCO pre-trained
epochs:         20
batch:          32
imgsz:          480               # input resolution
optimizer:      auto              # AdamW auto-selected
lr0:            0.01              # initial LR
lrf:            0.01              # final LR = lr0 × lrf
warmup_epochs:  3.0               # cosine warmup
amp:            true              # mixed precision FP16
```

### Augmentation Pipeline

| Augmentation | Value | Purpose |
|---|---|---|
| HSV Hue/Sat/Val | 0.015 / 0.7 / 0.4 | Lighting & weather variation |
| Horizontal Flip | 50% | Doubles effective dataset |
| Translate | 0.1 | Off-center plates |
| Scale | 0.5 | Distance variation |
| Mosaic | 1.0 | Partial plate learning |
| Random Erasing | 0.4 | Occlusion simulation |
| RandAugment | auto | Photometric diversity |
| Close Mosaic | Last 10 epochs | Stabilize fine-tuning |

### Loss Functions

```
L_total = 10.0 × L_box(CIoU) + 1.0 × L_cls(BCE) + 1.5 × L_dfl
```

- **Box Loss (CIoU)** — Weighted 10× because precise plate localization is critical for OCR
- **Classification Loss (BCE)** — Single-class plate vs. background
- **Distribution Focal Loss** — Allows uncertainty-aware boundary predictions

### Run Training

```bash
# Open the training notebook
jupyter notebook notebooks/train.ipynb
```

Or train via CLI:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=configs/knight_sight.yaml \
  epochs=20 \
  batch=32 \
  imgsz=480 \
  device=0
```

---

## 🔄 ONNX Export

Export the trained model to ONNX for optimized cross-platform inference:

```bash
python scripts/export_onnx.py
```

Or directly:

```python
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="onnx", simplify=True)
```

| Aspect | PyTorch (.pt) | ONNX (.onnx) |
|---|---|---|
| File Size | 5.3 MB | 10.5 MB |
| Inference Speed | ~8ms | ~6ms |
| Portability | Python only | C++, Java, mobile, browser |
| Further Optimization | CUDA | TensorRT, OpenVINO, CoreML |

---

## 📦 Creating Submissions

Generate competition submission files:

```bash
python create_submission.py
```

This produces the `Sapines_II/` folder containing:
- `predictions.json` — Plate bounding boxes for each test image
- `efficiency.json` — FLOPs, latency, and model size
- `efficiency_per_image_ms.json` — Per-image inference latency

The submission script uses a **3-tier detection strategy**:
1. Vehicle detection → plate detection in crop
2. Fallback: plate detection on full image
3. Fallback: 90° rotation + plate detection

---

## 🚀 Deployment

### Deploy to Hugging Face Spaces

```bash
# 1. Install the Hugging Face CLI
pip install huggingface_hub[cli]

# 2. Login to Hugging Face
huggingface-cli login

# 3. Create the Space (only first time)
huggingface-cli repo create deepsight-sapiens --type space --space-sdk streamlit

# 4. Add the HF remote
git remote add hf https://huggingface.co/spaces/Gyaanendra/deepsight-sapiens

# 5. Push to Hugging Face (triggers auto-deploy)
git push hf main
```

> **Note:** Hugging Face uses `packages.txt` for system-level apt dependencies and `requirements.txt` for Python packages. Both are already configured in this repo.

> **Environment Variables:** Set your `OPENROUTER_API_KEY` in the Space settings → Repository Secrets if you want the GPT-4o fallback to work on HF Spaces.

### Push to GitHub

```bash
# 1. Initialize Git LFS (if not already done)
git lfs install

# 2. Add the GitHub remote (first time only)
git remote add origin https://github.com/Gyaanendra/deepsight-sapiens.git

# 3. Stage all changes
git add .

# 4. Commit
git commit -m "feat: complete ANPR pipeline with dual OCR + ONNX support"

# 5. Push to GitHub
git push -u origin main
```

> **Important:** This repo uses Git LFS for large files (`.pt`, `.onnx`, `.pth`, `.ipynb`, `.jpg`, `.png`, `.pdf`, `.zip`). Make sure Git LFS is installed before pushing.

### Quick Deploy Script (Both)

```bash
# Push to both GitHub and HF Spaces in one go
git add .
git commit -m "update: latest changes"
git push origin main
git push hf main
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | No | API key for GPT-4o-mini OCR fallback via [OpenRouter](https://openrouter.ai) |

### Model Paths (in `app.py`)

| Constant | Path | Description |
|----------|------|-------------|
| `PT_PATH` | `models/best.pt` | Fine-tuned plate detector (PyTorch) |
| `ONNX_PATH` | `models/best.onnx` | Fine-tuned plate detector (ONNX) |
| `VMODEL` | `models/yolo11n.pt` | Vehicle detector (COCO pre-trained) |

### Vehicle Classes

The vehicle detector filters for COCO classes `[2, 3, 5, 7]`:
- `2` — Car
- `3` — Motorcycle
- `5` — Bus
- `7` — Truck

---

## 🔬 Technical Deep Dive

For a comprehensive technical report covering:
- YOLOv11 architecture (backbone, neck, head)
- Anchor-free detection mechanism
- Loss functions (CIoU, BCE, DFL) — with mathematical formulations
- Training curves analysis (epoch-by-epoch)
- Learning rate schedule (warmup + cosine decay)
- IoU, mAP, Precision, Recall — explained in depth
- ONNX export & edge optimization strategies

👉 **See [`explain.md`](explain.md)** — the full technical report.

---

## 🗺️ Future Roadmap

| Feature | Status | Details |
|---------|--------|---------|
| Multi-plate per vehicle | ✅ Done | Supported in current pipeline |
| ONNX inference | ✅ Done | Selectable from sidebar |
| GPU acceleration | ✅ Done | Auto CUDA detection |
| Video inference | ✅ Done | With configurable sample rate |
| Night vision (CLAHE) | ✅ Done | Toggle in sidebar |
| GPT-4o OCR fallback | ✅ Done | Via OpenRouter API |
| Live webcam stream | 🔲 Planned | `cv2.VideoCapture(0)` integration |
| Plate tracking (ByteTrack) | 🔲 Planned | Cross-frame plate tracking |
| Database logging | 🔲 Planned | SQLite with timestamps |
| Alert system | 🔲 Planned | Blocklist/whitelist plate lookup |
| TensorRT FP16 | 🔲 Planned | 2–3× faster on NVIDIA Jetson |
| OpenVINO INT8 | 🔲 Planned | Optimized for Intel edge CPUs |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'feat: add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/Gyaanendra/deepsight-sapiens.git
cd deepsight-sapiens
pip install -r requirements.txt
python -m pytest tests/       # run tests
streamlit run app.py          # launch dashboard
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [**Ultralytics**](https://ultralytics.com) — YOLOv11 framework
- [**Fast-Plate-OCR**](https://github.com/ankandrew/fast-plate-ocr) — Lightweight plate OCR engine
- [**OpenRouter**](https://openrouter.ai) — GPT-4o-mini API access for fallback OCR
- [**Streamlit**](https://streamlit.io) — Interactive dashboard framework
- [**Hugging Face**](https://huggingface.co) — Model hosting & Spaces deployment
- **KnightSight EdgeVision Challenge** — Competition organizers

---

<div align="center">

**Built with ❤️ for the KnightSight EdgeVision Challenge**

*Ultralytics YOLO · Fast-Plate-OCR · GPT-4o-mini · Streamlit · ONNX Runtime*

</div>
