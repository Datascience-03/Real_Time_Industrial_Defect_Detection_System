# Real-Time Industrial Defect Detection System

> **InspectAI** — A production-ready AI system for real-time detection of steel surface defects using YOLOv8 + ONNX Runtime, served via a FastAPI backend with a live interactive web dashboard.

---

## 🔍 Overview

This project detects 6 types of steel surface defects from both **images and MP4 videos** in real time. It includes:

- A **YOLOv8** model trained on the NEU Surface Defect dataset (~99.5% mAP@50)
- An **ONNX Runtime** inference backend for fast CPU/GPU inference
- A **FastAPI** REST API with health monitoring, Prometheus metrics, and PLC communication
- An **InspectAI web dashboard** for interactive defect detection via drag & drop
- **Containerization** via Docker with GPU (TensorRT) support
- Full **CI/CD** via GitHub Actions

---

## 🏷️ Defect Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `crazing` | Fine network of surface cracks |
| 1 | `inclusion` | Foreign material embedded in steel |
| 2 | `patches` | Irregular surface patches |
| 3 | `pitted_surface` | Small pits or holes on the surface |
| 4 | `rolled_in_scale` | Oxide scale rolled into the surface |
| 5 | `scratches` | Linear surface scratches |

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Datascience-03/Real_Time_Industrial_Defect_Detection_System.git
cd Real_Time_Industrial_Defect_Detection_System
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows
# or
source .venv/bin/activate      # Linux/macOS

pip install -r requirements.txt
```

### 3. Run the Web Dashboard

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🌐 Web Dashboard (InspectAI)

The dashboard allows you to:
- **Upload images** (JPG, PNG) and see defect bounding boxes drawn live on the canvas
- **Upload MP4 videos** and receive an annotated output video with detections, plus a per-class defect summary in the sidebar
- Monitor **System Health**, model status, and defect class legend in real time

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Serves the InspectAI dashboard |
| `/health` | `GET` | Returns system health and model status |
| `/predict` | `POST` | Upload a JPG/PNG image for defect detection |
| `/predict-video` | `POST` | Upload an MP4 video for frame-by-frame detection |
| `/plc/send` | `POST` | Receive defect data from a PLC controller |
| `/metrics` | `GET` | Prometheus metrics endpoint |
| `/docs` | `GET` | Swagger UI API documentation |

---

## 🚀 Usage

### Run Image Inference (Single Image)

```bash
python realtime_inference.py --source dataset/test/images/rolled-in_scale_277.jpg --no-display
```

### Run Video Defect Detection (Batch)

Processes all MP4 files in `dataset/videos/` and saves annotated outputs to `outputs/video_detection/`:

```bash
python src/video_detection.py
```

### Run Model Evaluation

Evaluates accuracy across the test set and generates precision-recall reports in `outputs/reports/`:

```bash
python run_evaluation.py
```

---

## 📁 Project Structure

```
├── src/
│   ├── app.py                  # Main FastAPI application & all API endpoints
│   ├── api.py                  # Standalone API (alternative entry point)
│   ├── video_detection.py      # Offline batch video processing script
│   ├── metrics.py              # Prometheus metrics definitions
│   ├── plc_sender.py           # PLC communication utilities
│   ├── evaluate.py             # Model evaluation pipeline
│   ├── video_capture.py        # OpenCV video capture helpers
│   └── utils.py                # Bounding box & utility functions
├── static/
│   └── index.html              # InspectAI web dashboard (single-file SPA)
├── dataset/
│   ├── train/                  # Training images & labels
│   ├── valid/                  # Validation images & labels
│   ├── test/                   # Test images & labels
│   └── videos/                 # Sample test videos (MP4)
├── runs/detect/train/weights/
│   ├── best.onnx               # Primary ONNX model (used for inference)
│   └── last.pt                 # Last checkpoint (PyTorch)
├── outputs/
│   ├── video_detection/        # Annotated output videos
│   └── reports/                # Evaluation reports & PR curves
├── docs/                       # Weekly reports and documentation
├── tests/integration/          # Integration test suite (pytest)
├── .github/workflows/          # GitHub Actions CI/CD pipelines
├── Dockerfile                  # CPU Docker build
├── Dockerfile.gpu              # GPU Docker build
├── docker-compose.yml          # Docker Compose (CPU)
├── docker-compose.gpu.yml      # Docker Compose (GPU)
└── requirements.txt            # Python dependencies
```

---

## 🧪 Running Tests

```bash
pytest tests/integration -v
```

All 9 integration tests cover: dashboard rendering, health check, image prediction validation, and smoke imports.

---

## 🐳 Docker Deployment

### CPU (Standard)

```bash
docker compose up -d
```

### GPU (NVIDIA Container Toolkit required)

```bash
docker compose -f docker-compose.gpu.yml build --pull
docker compose -f docker-compose.gpu.yml up -d
```

### TensorRT Engine Build (inside NVIDIA container)

**Bash:**
```bash
./scripts/build_trt_docker.sh nvcr.io/nvidia/tensorrt:23.09-py3 runs/detect/train/weights/best.onnx model.engine
```

**PowerShell:**
```powershell
.\scripts\build_trt_docker.ps1 -Image nvcr.io/nvidia/tensorrt:23.09-py3 -OnnxPath runs/detect/train/weights/best.onnx -EngineOut model.engine
```

> Note: You may need to `docker login nvcr.io` to pull NGC images.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model Architecture | YOLOv8n |
| Dataset | NEU Surface Defect |
| Classes | 6 |
| mAP@50 | ~99.5% |
| Runtime | ONNX Runtime (CPUExecutionProvider) |
| Inference Size | 640×640 |

---

## 📦 Release Artifacts

To package the ONNX model, pinned requirements, and Dockerfile into a release bundle:

**Bash:**
```bash
./scripts/package_release.sh
```

**PowerShell:**
```powershell
.\scripts\package_release.ps1
```

---

## 🔁 CI/CD

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `ci.yml` | Push/PR to `main` | Python syntax check + smoke test |
| `integration-and-trt-build.yml` | Push/PR to `main` | Full integration tests (CPU) |
| `trt-build-manual.yml` | Manual dispatch | TensorRT engine build on self-hosted GPU runner |

---

## 👥 Team

**Datascience-03** — Multi-member collaborative project

| Branch | Member |
|--------|--------|
| `Krishnakumar` | Krishnakumar J |
| `aashmika` | Aashmika |
| `manjesh` | Manjesh |
| `rukmani` | Rukmani |
| `visalam` | Visalam |
