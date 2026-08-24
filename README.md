# InspectAI — Real-Time Industrial Defect Detection System

A production-ready AI system for real-time detection of steel surface defects using **YOLOv8 + ONNX Runtime**, served through a **FastAPI backend** with an interactive **InspectAI web dashboard**, Prometheus monitoring, PLC communication, Docker deployment, and TensorRT GPU acceleration.

**Team:** Datascience-03
**Member:** Krishnakumar J
**Repository:** Datascience-03/Real_Time_Industrial_Defect_Detection_System
**Branch:** `Krishnakumar`
**Review Date:** 17 August 2026

---

## 🔍 Project Overview

InspectAI is a complete end-to-end industrial defect detection system designed to identify **6 types of steel surface defects** using a YOLOv8 deep learning model.

The system includes:

* **YOLOv8** model trained on the NEU Surface Defect Dataset
* **ONNX Runtime** inference for fast CPU inference
* **TensorRT FP16** engine for GPU-accelerated inference
* **FastAPI** backend with REST API endpoints
* **InspectAI Web Dashboard** for interactive image and video detection
* **Prometheus** metrics and monitoring
* **PLC / REST communication**
* **Docker** containerization with CPU and GPU support
* **GitHub Actions** CI/CD pipelines
* **Integration tests** and API benchmarking

### Key Performance

* Precision: **99.2%**
* Recall: **100%**
* mAP@50: **99.5%**
* TensorRT FP16 latency: **6.84 ms**
* TensorRT FP16 throughput: **146.15 FPS**
* GPU speedup: **10.13×**
* API benchmark success rate: **100%**

---

## 🏷️ Defect Classes

| ID | Class             | Description                         |
| -- | ----------------- | ----------------------------------- |
| 0  | `crazing`         | Fine network of surface cracks      |
| 1  | `inclusion`       | Foreign material embedded in steel  |
| 2  | `patches`         | Irregular surface patches           |
| 3  | `pitted_surface`  | Small pits or holes on the surface  |
| 4  | `rolled_in_scale` | Oxide scale rolled into the surface |
| 5  | `scratches`       | Linear surface scratches            |

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Datascience-03/Real_Time_Industrial_Defect_Detection_System.git
cd Real_Time_Industrial_Defect_Detection_System
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🌐 Web Dashboard

InspectAI provides a web dashboard for interactive defect detection.

The dashboard supports:

* Uploading JPG and PNG images
* Detecting defects with bounding boxes
* Uploading MP4 videos
* Frame-by-frame video defect detection
* Displaying per-class defect summaries
* Confidence threshold adjustment
* System health monitoring
* Model status monitoring
* Defect class visualization

## Start the Dashboard

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

For development with automatic reload:

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000 --reload
```

---

# 📡 API Endpoints

| Endpoint         | Method | Description                                      |
| ---------------- | ------ | ------------------------------------------------ |
| `/`              | `GET`  | Serves the InspectAI dashboard                   |
| `/health`        | `GET`  | Returns system health and model status           |
| `/predict`       | `POST` | Upload a JPG/PNG image for defect detection      |
| `/predict-video` | `POST` | Upload an MP4 video for frame-by-frame detection |
| `/plc/send`      | `POST` | Receives defect data from a PLC controller       |
| `/metrics`       | `GET`  | Prometheus metrics endpoint                      |
| `/docs`          | `GET`  | Swagger UI API documentation                     |

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

# 📅 Week-by-Week Progress

## Week 1 — Dataset Preparation

* Downloaded and organized the **NEU Surface Defect Dataset**
* Dataset contains **6 defect classes**
* Split dataset into:

  * **80% Training**
  * **10% Validation**
  * **10% Testing**
* Verified dataset integrity
* Found zero corrupted images
* Generated YOLO-format annotation labels

---

## Week 2 — Model Training

The YOLOv8n model was trained using:

* Model: **YOLOv8n**
* Epochs: **20**
* Image size: **640**
* Batch size: **16**

### Training Results

| Metric    | Result    |
| --------- | --------- |
| Precision | **99.2%** |
| Recall    | **100%**  |
| mAP@50    | **99.5%** |

---

## Week 3 — Export & Real-Time Inference

The trained model was exported to **ONNX** and compiled into a **TensorRT FP16 engine** for GPU acceleration.

The system also includes an OpenCV-based video inference pipeline.

### Video Processing

* Videos processed: **3**
* Total frames processed: **1,494**
* Total detections: **460**

### Runtime Benchmark

| Runtime           | Device       | Latency     | FPS        | Speedup    |
| ----------------- | ------------ | ----------- | ---------- | ---------- |
| PyTorch CPU       | CPU          | 69.32 ms    | 14.43      | 1.00×      |
| ONNX Runtime      | CPU          | 61.73 ms    | 16.20      | 1.12×      |
| **TensorRT FP16** | **RTX 3050** | **6.84 ms** | **146.15** | **10.13×** |

TensorRT FP16 provides a **10.13× speedup** compared with the PyTorch CPU baseline.

---

## Week 4 — API, Dashboard, Monitoring & Deployment

Implemented:

* FastAPI backend
* Image prediction endpoint
* Video prediction endpoint
* Health monitoring
* PLC REST simulation
* Prometheus metrics
* InspectAI web dashboard
* Drag-and-drop upload interface
* Confidence threshold slider
* Live bounding boxes
* Video detection summary
* Docker containerization
* CPU and GPU deployment configurations
* API benchmarking
* Integration testing

### API Benchmark

The API was benchmarked using:

* **100 requests**
* **Concurrency: 10**

All tested endpoints achieved a **100% success rate**.

| Endpoint         | Requests/sec | Average Latency |
| ---------------- | ------------ | --------------- |
| `GET /health`    | 336.69       | 27.57 ms        |
| `POST /plc/send` | 301.95       | 31.25 ms        |
| `POST /predict`  | 12.78        | 758.19 ms       |

---

# 🧪 Demonstration Commands

## Run Image Inference

```powershell
python realtime_inference.py --source dataset/test/images/rolled_in_scale_277.jpg --no-display
```

---

## Run Video Defect Detection

Processes MP4 files from `dataset/videos/` and saves annotated outputs to `outputs/video_detection/`.

```bash
python src/video_detection.py
```

---

## Run Model Evaluation

```bash
python run_evaluation.py
```

---

## Run Integration Tests

The integration test suite contains **9 tests**.

```bash
python -m pytest tests/integration -v
```

Expected result:

```text
9 passed
```

---

## Run API Benchmark

```bash
python src/api_benchmark.py
```

---

# 📁 Project Structure

```text
Real_Time_Industrial_Defect_Detection_System/
│
├── src/
│   ├── app.py                  # Main FastAPI application and API endpoints
│   ├── api.py                  # Standalone API alternative entry point
│   ├── video_detection.py      # Offline batch video processing
│   ├── metrics.py              # Prometheus metrics definitions
│   ├── plc_sender.py           # PLC communication utilities
│   ├── evaluate.py             # Model evaluation pipeline
│   ├── video_capture.py        # OpenCV video capture helpers
│   └── utils.py                # Bounding box and utility functions
│
├── static/
│   └── index.html               # InspectAI web dashboard
│
├── dataset/
│   ├── train/                   # Training images and labels
│   ├── valid/                   # Validation images and labels
│   ├── test/                    # Test images and labels
│   └── videos/                  # Sample test videos
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.onnx    # Primary ONNX model
│               └── last.pt      # Last PyTorch checkpoint
│
├── outputs/
│   ├── video_detection/         # Annotated output videos
│   └── reports/                 # Evaluation reports and PR curves
│
├── docs/                        # Weekly reports and documentation
│
├── tests/
│   └── integration/             # Integration test suite
│
├── .github/
│   └── workflows/               # GitHub Actions CI/CD pipelines
│
├── Dockerfile                   # CPU Docker build
├── Dockerfile.gpu               # GPU Docker build
├── docker-compose.yml           # CPU Docker Compose configuration
├── docker-compose.gpu.yml       # GPU Docker Compose configuration
└── requirements.txt             # Python dependencies
```

---

# 🐳 Docker Deployment

## CPU Deployment

Build and start the CPU container:

```bash
docker compose up -d
```

---

## GPU Deployment

NVIDIA Container Toolkit is required.

Build the GPU container:

```bash
docker compose -f docker-compose.gpu.yml build --pull
```

Start the GPU deployment:

```bash
docker compose -f docker-compose.gpu.yml up -d
```

---

# ⚡ TensorRT Engine Build

TensorRT FP16 can be built using the provided Docker scripts.

## Bash

```bash
./scripts/build_trt_docker.sh nvcr.io/nvidia/tensorrt:23.09-py3 runs/detect/train/weights/best.onnx model.engine
```

## PowerShell

```powershell
.\scripts\build_trt_docker.ps1 -Image nvcr.io/nvidia/tensorrt:23.09-py3 -OnnxPath runs/detect/train/weights/best.onnx -EngineOut model.engine
```

> **Note:** You may need to authenticate with NVIDIA NGC before pulling the TensorRT image.

```bash
docker login nvcr.io
```

---

# ⚠️ Challenges & Resolutions

| Challenge                                                     | Resolution                                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| ONNX model loaded 80 COCO classes instead of 6 defect classes | Re-exported from the correct project `best.pt` weights                         |
| OpenCV display error in headless environments                 | Disabled live window and saved annotated output videos                         |
| TensorRT Python bindings were not found after installation    | Created a TensorRT wrapper redirecting to `tensorrt_bindings`                  |
| `model.engine` exceeded GitHub's 100 MB file limit            | Excluded the large engine from normal Git tracking and handled it separately   |
| GitHub Actions GPU runner was unavailable                     | Decoupled TensorRT build into a separate manual workflow                       |
| MP4 video codec was not browser-compatible                    | Switched video output to H.264 / `avc1`                                        |
| Video detection sidebar was blank                             | Added custom HTTP headers to return per-class detection counts to the frontend |

---

# 🔁 CI/CD Pipeline

| Workflow              | Trigger                   | Status    |
| --------------------- | ------------------------- | --------- |
| CI / checks           | Every push                | ✅ Passing |
| CPU Integration Tests | Every push / Pull Request | ✅ Passing |
| TensorRT GPU Build    | Manual trigger            | ✅ Ready   |

---

# 🎯 Key Achievements

1. **10.13× GPU speedup** — TensorRT FP16 achieved 146 FPS and significantly exceeded the 30 FPS industrial line requirement.

2. **99.5% mAP@50** — The trained model achieved high detection accuracy across all 6 defect classes.

3. **100% API stability** — The tested API endpoints maintained a 100% success rate under 10 concurrent clients.

4. **Full-stack delivery** — The project covers the complete pipeline from dataset preparation and model training to GPU inference, REST API, web dashboard, monitoring, Docker deployment, and CI/CD.

5. **Browser-compatible video detection** — Annotated videos are encoded using H.264 and can be played directly in the browser.

---

# 🧠 Key Technical Talking Points

### 1. GPU Acceleration

TensorRT FP16 reduced inference latency from:

```text
69.32 ms → 6.84 ms
```

and increased throughput from:

```text
14.43 FPS → 146.15 FPS
```

resulting in a:

```text
10.13× speedup
```

---

### 2. Model Localization Limitation

The NEU dataset used in this project primarily provides whole-image defect labels.

For more precise defect localization, fine-tuning with **NEU-DET** or another dataset containing precise bounding-box annotations is recommended as a future improvement.

---

### 3. API Stability

The API benchmark tested the system using:

```text
100 requests
Concurrency: 10
```

and achieved a **100% success rate** across the tested endpoints.

---

### 4. Production Deployment Path

A scalable factory deployment can use:

```text
Camera / Video Stream
        ↓
YOLOv8 Model
        ↓
TensorRT FP16
        ↓
FastAPI
        ↓
InspectAI Dashboard
        ↓
PLC / Monitoring System
```

For multi-stream deployments, TensorRT GPU inference combined with multiple Uvicorn workers can be considered.

---

### 5. Full-Stack System

The final system integrates:

```text
Dataset
   ↓
YOLOv8 Training
   ↓
ONNX Export
   ↓
TensorRT FP16
   ↓
Real-Time Inference
   ↓
FastAPI
   ↓
InspectAI Web Dashboard
   ↓
Prometheus Monitoring
   ↓
PLC Communication
   ↓
Docker
   ↓
GitHub Actions CI/CD
```

---

# 👥 Final Team Status

| Member           | Task                                                        | Status |
| ---------------- | ----------------------------------------------------------- | ------ |
| Rukmani Priya    | FastAPI Backend & Video Detection                           | ✅ Done |
| Visalam          | PLC / External REST Communication                           | ✅ Done |
| Aashmika         | Prometheus Metrics & Logging                                | ✅ Done |
| Manjesh          | Dockerization & Container Setup                             | ✅ Done |
| **Krishnakumar** | Benchmarking, TensorRT, CI/CD, Integration Tests, Bug Fixes | ✅ Done |

> [!IMPORTANT]
> All source code is committed, tested, and pushed to `origin/Krishnakumar`. The repository is fully clean and review-ready.

---

# 📌 Project Status

**Status: Completed and Review-Ready**

The project successfully integrates machine learning inference, GPU acceleration, backend APIs, web visualization, monitoring, PLC communication, containerization, testing, benchmarking, and CI/CD into a complete real-time industrial defect detection system.
