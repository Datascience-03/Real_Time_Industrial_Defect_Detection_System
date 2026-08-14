# Real-Time Industrial Defect Detection System

## Project Completion Summary

This project is a complete real-time industrial defect detection system for steel surface defects. It includes dataset preparation, YOLOv8 model training, ONNX/TensorRT export, real-time inference, API integration, monitoring, containerization, PLC communication, and performance benchmarking.

## What was completed

- **Dataset preparation and preprocessing**
  - Organized the NEU surface defect dataset into 6 classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.
  - Split data into training, validation, and test sets.
  - Built a preprocessing pipeline to resize and normalize images for YOLO training.
  - Generated YOLO-format labels for the defect classes.

- **Model training and validation**
  - Trained a YOLOv8n object detection model using the prepared dataset.
  - Produced model weights (`best.pt`, `last.pt`) and training artifacts.
  - Achieved strong detection performance with approximately 99.5% mAP@50.

- **Model export and verification**
  - Exported the trained YOLO model to ONNX format.
  - Verified ONNX model correctness and class labels.
  - Converted the ONNX model into a TensorRT engine for GPU-accelerated inference.

- **Real-time inference and video detection**
  - Built an OpenCV-based video capture and preprocessing pipeline.
  - Integrated the inference model for real-time defect detection on videos.
  - Generated annotated output videos showing detected defects and confidence scores.
  - Verified real-time performance and inference stability.

- **Backend API integration**
  - Implemented FastAPI endpoints for system health, prediction, and PLC data communication.
  - Added `/health`, `/predict`, and `/plc/send` endpoints.
  - Confirmed successful REST API payload handling with HTTP 200 responses.

- **Monitoring and logging**
  - Added Prometheus metrics support for request counts, latency, uptime, and inference FPS.
  - Exposed metrics via the `/metrics` endpoint.
  - Implemented application logging for API calls and prediction events.

- **Containerization and deployment**
  - Created a `Dockerfile` for reproducible container builds.
  - Added `docker-compose.yml` for deploying the FastAPI service.
  - Packaged dependencies, model files, and source code for edge deployment.

- **API benchmarking and performance evaluation**
  - Developed an API benchmarking suite (`src/api_benchmark.py`).
  - Tested `/health`, `/predict`, and `/plc/send` under concurrent load.
  - Produced a final performance report in `docs/week4_performance_metrics.txt`.
  - Confirmed 100% endpoint success rates and stable latency.

## Final system status

- All major tasks are marked as complete in the project reports.
- Week 4 report confirms final system completion with all member tasks done.
- The system is ready for production validation with GPU-accelerated inference and API deployment.

## Key deliverables

- `src/app.py`, `src/api.py`, `src/plc_sender.py`
- `src/export_onnx.py`, `src/verify_onnx.py`, `build_trt_engine.py`
- `Dockerfile`, `docker-compose.yml`
- `docs/week4_Report.md`, `docs/week4_performance_metrics.txt`
- `outputs/video_detection/`

---

## Notes

- The final reports describe successful completion of dataset preparation, model training, export, real-time video detection, API integration, monitoring, containerization, and benchmark validation.
- For a public project overview, this README now captures the completed state and major achievements.

## Quick usage

Run realtime inference (image):

```bash
python realtime_inference.py --source dataset/test/images/rolled-in_scale_277.jpg --no-display
```

Run evaluation across test set (produces reports in `outputs/reports`):

```bash
python run_evaluation.py
```

Export and build TensorRT engine (host or NVIDIA container)

1. Export ONNX if you don't have one:

```bash
python -c "from ultralytics import YOLO; YOLO('runs/detect/train/weights/best.pt').export(format='onnx')"
```

2a. Preferred: inside an NVIDIA container with TensorRT, run `trtexec`:

```bash
./scripts/build_trt.sh runs/detect/train/weights/best.onnx model.engine
```

2b. Or attempt via Python bindings (only possible if TensorRT is installed on host):

```bash
python build_trt_engine.py
```

Notes: If the system lacks TensorRT, `build_trt_engine.py` will fallback to `trtexec` if present, otherwise it will print instructions for installing TensorRT or using an NVIDIA container.

## Container (GPU)

Build and run GPU container (requires NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.gpu.yml build --pull
docker compose -f docker-compose.gpu.yml up -d
```

CI
- A lightweight GitHub Actions workflow is included at `.github/workflows/ci.yml` that checks Python syntax and runs a smoke script. The workflow intentionally avoids installing heavy ML packages.

Additional resources added
- `Dockerfile.gpu.trt` — a TensorRT-ready Dockerfile (based on NVIDIA NGC TensorRT image). Note: NGC images may require authentication.
- `scripts/pin_requirements.sh` — creates a virtualenv and writes `requirements-pinned.txt` via `pip freeze`.
- `requirements-pinned-template.txt` — template and instructions for a pinned requirements file.
- `notebooks/demo_inference.ipynb` — a small demo notebook that runs inference on a sample test image and saves an annotated result.

TensorRT build (containerized)

If you cannot install TensorRT on the host, build the engine inside an NVIDIA TensorRT container using the helper scripts:

Bash:
```bash
./scripts/build_trt_docker.sh nvcr.io/nvidia/tensorrt:23.09-py3 runs/detect/train/weights/best.onnx model.engine
```

PowerShell:
```powershell
.\scripts\build_trt_docker.ps1 -Image nvcr.io/nvidia/tensorrt:23.09-py3 -OnnxPath runs/detect/train/weights/best.onnx -EngineOut model.engine
```

Notes: you may need to `docker login nvcr.io` to pull NGC images. If you cannot access NGC, use a local CUDA/TensorRT-enabled image that includes `trtexec`.

CI and Release

- There is a GitHub Actions workflow at `.github/workflows/integration-and-trt-build.yml` that runs lightweight CPU tests and will attempt a TensorRT engine build on a self-hosted runner labeled `gpu`.
- To create release artifacts (ONNX, model.engine, pinned requirements and Dockerfile), run:

```bash
./scripts/package_release.sh
```

or on Windows PowerShell:

```powershell
.\scripts\package_release.ps1
```




