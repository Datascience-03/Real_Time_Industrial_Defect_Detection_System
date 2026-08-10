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
