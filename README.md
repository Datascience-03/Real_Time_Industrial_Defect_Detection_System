# Final Project Review — Real-Time Industrial Defect Detection System

**Team:** Datascience-03 | **Member:** Krishnakumar J  
**Repository:** [Datascience-03/Real_Time_Industrial_Defect_Detection_System](https://github.com/Datascience-03/Real_Time_Industrial_Defect_Detection_System)  
**Branch:** `Krishnakumar` | **Review Date:** 17 August 2026  

---

## 1. Project Overview

A complete end-to-end real-time industrial defect detection system that identifies **6 types of steel surface defects** using a YOLOv8 deep learning model, served through a FastAPI backend with a live web dashboard, Prometheus monitoring, PLC integration, and GPU-accelerated TensorRT inference.

**Defect Classes:** Crazing · Inclusion · Patches · Pitted Surface · Rolled-in Scale · Scratches

---

## 2. Week-by-Week Progress

### Week 1 — Dataset Preparation
- Downloaded and organized the **NEU Surface Defect Dataset** (6 classes).
- Split: **80% Train / 10% Validation / 10% Test**.
- Verified integrity — zero corrupted images found.
- Generated YOLO-format annotation labels for all 6 defect classes.

### Week 2 — Model Training
- Trained **YOLOv8n** for 20 epochs (image size 640, batch 16).
- **Precision: 99.2% · Recall: 100% · mAP@50: 99.5%**

### Week 3 — Export & Real-Time Inference
- Exported to **ONNX** and compiled **TensorRT FP16 engine** on NVIDIA RTX 3050.
- Built OpenCV video pipeline. Processed 3 videos: 1,494 frames, 460 detections.

| Runtime | Device | Latency | FPS | Speedup |
|---------|--------|---------|-----|---------|
| PyTorch CPU | CPU | 69.32 ms | 14.43 | 1.0x |
| ONNX Runtime | CPU | 61.73 ms | 16.20 | 1.12x |
| **TensorRT FP16** | **RTX 3050** | **6.84 ms** | **146.15** | **10.13x** |

### Week 4 — API, Dashboard, Monitoring & Deployment
- FastAPI backend: `/health`, `/predict`, `/plc/send`, `/metrics`
- InspectAI Web Dashboard with drag-drop, confidence slider, live bounding boxes
- Prometheus monitoring, PLC REST simulation, Docker containerization
- API Benchmark (100 req, concurrency 10): **100% success rate** on all endpoints

| Endpoint | Concurrency 10 RPS | Avg Latency |
|----------|-------------------|-------------|
| GET /health | 336.69 | 27.57 ms |
| POST /plc/send | 301.95 | 31.25 ms |
| POST /predict | 12.78 | 758.19 ms |

---

## 3. Demonstration Commands

```powershell
# Start the Web Dashboard
uvicorn src.app:app --host 127.0.0.1 --port 8000
# Open: http://127.0.0.1:8000

# Run real-time inference on a test image
python realtime_inference.py --source dataset/test/images/rolled-in_scale_277.jpg --no-display

# Run all integration tests (9/9 passing)
python -m pytest tests/integration -v

# Run API benchmark
python src/api_benchmark.py
```

---

## 4. Challenges & Resolutions

| Challenge | Resolution |
|-----------|------------|
| ONNX model loaded 80 COCO classes instead of 6 defect classes | Re-exported from the correct project `best.pt` weights |
| OpenCV display error in headless environments | Disabled live window; saved annotated output video files |
| TensorRT Python bindings not found after `pip install` (Python 3.14 packaging bug) | Manually created `tensorrt/__init__.py` wrapper redirecting to `tensorrt_bindings` |
| `model.engine` (164 MB) rejected by GitHub (100 MB limit) | Excluded via `git update-index --assume-unchanged`; uploaded via manual workflow artifact |
| GitHub Actions GPU runner offline | Decoupled TensorRT build to a separate `workflow_dispatch` manual workflow |

---

## 5. CI/CD Pipeline Status

| Workflow | Trigger | Status |
|----------|---------|--------|
| CI / checks | Every push | ✅ Passing |
| Tests — CPU integration (9 tests) | Every push/PR | ✅ Passing |
| TensorRT GPU Build | Manual trigger | ✅ Ready |

---

## 6. Key Talking Points for Review

1. **10.13x GPU speedup** — TensorRT FP16 at 146 FPS far exceeds the 30 FPS industrial line requirement.
2. **Model localization limitation** — NEU dataset uses whole-image labels; fine-tuning on NEU-DET with precise bounding boxes is the recommended next step.
3. **API stability** — 100% success rate maintained under 10 concurrent clients in benchmarking.
4. **Production path** — Replace CPU ONNX inference with TensorRT + `uvicorn --workers N` for scalable multi-stream factory deployment.
5. **Full stack delivered** — From raw dataset to trained model to GPU inference to REST API to web UI to CI/CD pipeline.

---

## 7. Final Team Status

| Member | Task | Status |
|--------|------|--------|
| Rukmani Priya | FastAPI Backend & Video Detection | ✅ Done |
| Visalam | PLC / External REST Communication | ✅ Done |
| Aashmika | Prometheus Metrics & Logging | ✅ Done |
| Manjesh | Dockerization & Container Setup | ✅ Done |
| **Krishnakumar** | Benchmarking, TensorRT, CI/CD, Integration Tests | ✅ Done |

> [!IMPORTANT]
> All source code is committed, tested, and pushed to `origin/Krishnakumar`. The repository is fully clean and review-ready.
