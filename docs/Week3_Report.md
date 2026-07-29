# Week 3 Final Report

**Project:** Real-Time Industrial Defect Detection System  
**Timeline:** Days 1 - 7 (Week 3)  
**Status:** Completed  

---

## 1. Executive Summary
Week 3 focused on moving from a trained model to a real-time edge deployment pipeline. The objectives were to export the YOLOv8 model to optimized formats (ONNX, TensorRT), build a frame preprocessing and video capture pipeline, integrate the model into a live stream, and perform thorough performance evaluations and error analysis.

All targets have been met. The system successfully runs real-time inference on high-resolution industrial videos, achieving a throughput of **146.15 FPS** with hardware acceleration (TensorRT on GPU), exceeding standard industrial processing speed requirements.

---

## 2. Work Completed (Task & Member Distribution)

### 🔹 Member 1 (Rukmani Priya & Visalam): Model Export
* **Task:** Export trained YOLOv8 model weights (`best.pt`) to ONNX and verify correctness.
* **Deliverables:** `src/export_onnx.py`, `src/verify_onnx.py`, `runs/detect/train/weights/best.onnx`, export documentation.
* **Result:** Model was successfully exported to ONNX format. Correctness was verified by running inference on test partition images, yielding predictions consistent with the PyTorch model.

### 🔹 Member 2 (Manjesh): TensorRT Engine Compilation
* **Task:** Convert ONNX model to TensorRT Engine and run baseline benchmarks.
* **Deliverables:** `build_trt_engine.py`, `model.engine`, TensorRT benchmark results.
* **Result:** Compiled a native TensorRT engine `model.engine` optimized for FP16 precision. Achieved extremely low latency on an NVIDIA GeForce RTX 3050.

### 🔹 Member 3 (Aashmika): OpenCV Video Capture Pipeline
* **Task:** Create OpenCV capture pipeline (video input, resizing, frame handling).
* **Deliverables:** `src/video_capture.py`, frame preprocessing pipeline.
* **Result:** Developed a robust frame acquisition module that handles video files or webcam streams, resizes frames to `640x640` to match YOLO requirements, and manages memory resources.

### 🔹 Member 4 (Manjesh & Aashmika): Real-Time Integration
* **Task:** Integrate TensorRT/ONNX model with OpenCV for real-time defect detection.
* **Deliverables:** `dataset/augmented/realtime_inference.py`, real-time FPS overlay, drawing bounding boxes.
* **Result:** Integrated the inference engine with the OpenCV video loop. Detections are plotted on-the-fly, and the active frame rate is displayed dynamically on the screen.

### 🔹 Member 5 (Krishnakumar): Performance Evaluation & Documentation
* **Task:** Performance testing, quantitative benchmarking, qualitative error analysis, and final reporting.
* **Deliverables:** `src/benchmark.py`, `src/generate_demo.py`, unified performance metrics report (`docs/performance_metrics.txt`), demo video (`outputs/demo_video.mp4`), sample screenshots, qualitative error analysis report (`docs/Week3_Error_Analysis_Report.md`), and final Week 3 Report.
* **Result:** Benchmarked the PyTorch CPU, ONNX CPU, and TensorRT GPU runtimes. Generated a complete demo video with defect box overlays and real-time FPS display, and documented qualitative limitations and recommended adjustments.

---

## 3. Quantitative Performance Comparison
We measured the inference latency and throughput (FPS) of the defect detection model across different runtimes. CPU benchmarks were run on the deployment machine, and GPU benchmarks were run on the NVIDIA GeForce RTX 3050.

| Model / Runtime Environment | Device | Avg Latency (ms) | Throughput (FPS) | Speedup vs PyTorch CPU |
| :--- | :--- | :---: | :---: | :---: |
| **PyTorch (Native)** | CPU | 69.32 ms | 14.43 FPS | 1.00x (Baseline) |
| **ONNX Runtime (Optimized)** | CPU | 61.73 ms | 16.20 FPS | 1.12x |
| **TensorRT (Compiled FP16)** | NVIDIA RTX 3050 GPU | **6.84 ms** | **146.15 FPS** | **10.13x** |

### Key Performance Findings:
* **ONNX Runtime** provides a minor **12% throughput improvement** on CPU.
* **TensorRT (NVIDIA GPU)** delivers a massive **10.13x speedup** over PyTorch CPU.
* Native PyTorch or ONNX on standard CPUs is insufficient for real-time factory floor lines (typically requiring 30+ FPS). However, GPU-accelerated TensorRT inference runs at **146.15 FPS**, which easily accommodates high-speed steel rolling mill lines.

---

## 4. Qualitative Error Analysis Summary
A detailed qualitative error analysis was conducted on the generated video detections. The primary findings are:

1. **Centered Bounding Box Bias:** The training dataset (NEU) was a pre-cropped classification dataset, and training boxes were programmatically generated as centered boxes covering 90% of the image. Consequently, on the test video stream, the model tends to draw a single giant centered bounding box around defects, rather than precise, localized boundaries.
2. **False Positives:** Under shifting lighting conditions, normal surface grain or scale textures are occasionally flagged as minor defects (crazing or rolled-in scale) with low confidence (30%-45%).
3. **Mitigation Recommendation:** The system must be fine-tuned using a true localization dataset (e.g., NEU-DET) with hand-annotated tight bounding boxes, and operational confidence thresholds should be adjusted to 0.50.

---

## 5. Deliverables & Outputs Location
* **Benchmarking Script:** [benchmark.py](file:///d:/Real_Time_Industrial_Defect_Detection_System/src/benchmark.py)
* **Demo Stream Script:** [generate_demo.py](file:///d:/Real_Time_Industrial_Defect_Detection_System/src/generate_demo.py)
* **Performance Metric Report:** [performance_metrics.txt](file:///d:/Real_Time_Industrial_Defect_Detection_System/docs/performance_metrics.txt)
* **Error Analysis Report:** [Week3_Error_Analysis_Report.md](file:///d:/Real_Time_Industrial_Defect_Detection_System/docs/Week3_Error_Analysis_Report.md)
* **Annotated Demo Video:** `outputs/demo_video.mp4`
* **Defect Detections (Screenshots):** `outputs/sample_results/screenshot_1.jpg`, `screenshot_2.jpg`, `screenshot_3.jpg`
