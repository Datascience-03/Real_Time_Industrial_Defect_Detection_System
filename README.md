# Real-Time Industrial Defect Detection System

> **InspectAI** — A production-ready AI system for real-time detection of steel surface defects using YOLOv8 + ONNX Runtime, served via a FastAPI backend with a live interactive web dashboard.

**Team:** Datascience-03 | **Repository:** [Datascience-03/Real_Time_Industrial_Defect_Detection_System](https://github.com/Datascience-03/Real_Time_Industrial_Defect_Detection_System)

---

## 🔍 Overview

A complete end-to-end system that identifies **6 types of steel surface defects** using a YOLOv8 deep learning model. It includes:

- A **YOLOv8** model trained on the NEU Surface Defect dataset (~99.5% mAP@50)
- An **ONNX Runtime** inference backend for fast CPU inference
- A **TensorRT FP16** engine for GPU-accelerated inference (10.13× speedup)
- A **FastAPI** REST API with health monitoring, Prometheus metrics, and PLC communication
- An **InspectAI web dashboard** for interactive image & video defect detection
- **Containerization** via Docker with GPU support
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

Open your browser at **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🌐 Web Dashboard (InspectAI)

The dashboard allows you to:
- **Upload images** (JPG, PNG) and see defect bounding boxes drawn live on the canvas
- **Upload MP4 videos** and receive an annotated output video with a per-class defect summary in the sidebar
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

### Run the Web Dashboard

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
# Open: http://127.0.0.1:8000
```

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

```bash
python run_evaluation.py
```

### Run Integration Tests (9/9 passing)

```bash
python -m pytest tests/integration -v
```

### Run API Benchmark

```bash
python src/api_benchmark.py
```

---

## 📅 Week-by-Week Progress

### Week 1 — Dataset Preparation
- Downloaded and organized the **NEU Surface Defect Dataset** (6 classes)
- Split: **80% Train / 10% Validation / 10% Test**
- Verified integrity — zero corrupted images found
- Generated YOLO-format annotation labels for all 6 defect classes

### Week 2 — Model Training
- Trained **YOLOv8n** for 20 epochs (image size 640, batch 16)
- **Precision: 99.2% · Recall: 100% · mAP@50: 99.5%**

### Week 3 — Export & Real-Time Inference
- Exported to **ONNX** and compiled **TensorRT FP16 engine** on NVIDIA RTX 3050
- Built OpenCV video pipeline. Processed 3 videos: 1,494 frames, 460 detections

| Runtime | Device | Latency | FPS | Speedup |
|---------|--------|---------|-----|---------|
| PyTorch CPU | CPU | 69.32 ms | 14.43 | 1.0× |
| ONNX Runtime | CPU | 61.73 ms | 16.20 | 1.12× |
| **TensorRT FP16** | **RTX 3050** | **6.84 ms** | **146.15** | **10.13×** |

### Week 4 — API, Dashboard, Monitoring & Deployment
- FastAPI backend: `/health`, `/predict`, `/predict-video`, `/plc/send`, `/metrics`
- InspectAI Web Dashboard with drag-drop, confidence slider, live bounding boxes, video detection summary
- Prometheus monitoring, PLC REST simulation, Docker containerization
- API Benchmark (100 req, concurrency 10): **100% success rate** on all endpoints

| Endpoint | Concurrency 10 RPS | Avg Latency |
|----------|-------------------|-------------|
| GET /health | 336.69 | 27.57 ms |
| POST /plc/send | 301.95 | 31.25 ms |
| POST /predict | 12.78 | 758.19 ms |

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

## ⚠️ Challenges & Resolutions

| Challenge | Resolution |
|-----------|------------|
| ONNX model loaded 80 COCO classes instead of 6 defect classes | Re-exported from the correct project `best.pt` weights |
| OpenCV display error in headless environments | Disabled live window; saved annotated output video files |
| TensorRT Python bindings not found after `pip install` (Python 3.14 bug) | Manually created `tensorrt/__init__.py` wrapper redirecting to `tensorrt_bindings` |
| `model.engine` (164 MB) rejected by GitHub (100 MB limit) | Excluded via `git update-index --assume-unchanged`; uploaded via manual workflow artifact |
| GitHub Actions GPU runner offline | Decoupled TensorRT build to a separate `workflow_dispatch` manual workflow |
| Video codec `mp4v` not playable in browser | Switched to `avc1` (H.264) codec in `VideoWriter` |
| Video detection sidebar blank after analysis | Added custom HTTP headers to return per-class detection counts to the frontend |

---

## 🔁 CI/CD Pipeline

| Workflow | Trigger | Status |
|----------|---------|--------|
| CI / checks | Every push to `main` | ✅ Passing |
| Tests — CPU integration (9 tests) | Every push/PR | ✅ Passing |
| TensorRT GPU Build | Manual trigger | ✅ Ready |

---

## 🎯 Key Achievements

1. **10.13× GPU speedup** — TensorRT FP16 at 146 FPS far exceeds the 30 FPS industrial line requirement
2. **99.5% mAP@50** — near-perfect detection accuracy across all 6 defect classes
3. **100% API stability** — maintained under 10 concurrent clients in benchmarking
4. **Full-stack delivery** — from raw dataset to trained model → GPU inference → REST API → web UI → CI/CD pipeline
5. **Browser-compatible video detection** — H.264 encoded annotated videos playable directly in the browser

---

## 👥 Team

| Member | Task | Status |
|--------|------|--------|
| Rukmani Priya | FastAPI Backend & Video Detection | ✅ Done |
| Visalam | PLC / External REST Communication | ✅ Done |
| Aashmika | Prometheus Metrics & Logging | ✅ Done |
| Manjesh | Dockerization & Container Setup | ✅ Done |
| **Krishnakumar** | Benchmarking, TensorRT, CI/CD, Integration Tests, Bug Fixes | ✅ Done |

> [!IMPORTANT]
> All source code is committed, tested, and pushed. The repository is fully clean and review-ready.
