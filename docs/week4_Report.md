Week 4 
## Report – Member 1

To integrate the trained YOLO defect detection model with a FastAPI backend, verify the ONNX model, test image prediction, and perform video-based defect detection.

### Work Completed
Integrated the trained YOLO model with FastAPI.
Implemented /health and /predict API endpoints.
Tested image prediction using Swagger UI.
Exported the trained model from .pt to ONNX.
Verified that the ONNX model contains the required 6 defect classes:
Crazing
Inclusion
Patches
Pitted Surface
Rolled-in Scale
Scratches
Evaluated the model on the test dataset.
Implemented video-based defect detection.
Successfully processed 3 videos and generated annotated output videos.
Model Evaluation
Metric	Result
Precision	99.2%
Recall	100%
mAP@50	99.5%
mAP@50-95	99.5%
Video Detection Results
Video	Frames	Detections
sample1.mp4	241	92
sample2.mp4	901	243
sample3.mp4	352	125
Total	1,494	460
Output

The processed videos were saved in:

outputs/video_detection/

with:

sample1_detected.mp4
sample2_detected.mp4
sample3_detected.mp4
Challenges

An incorrect ONNX model initially showed COCO 80 classes. It was replaced with the correctly exported project model containing the required six industrial defect classes. An OpenCV display error was also resolved by processing videos without live window display.

### Conclusion

Week 4 successfully completed the backend API integration, ONNX verification, image prediction testing, and multi-video defect detection. The system successfully generated annotated videos with defect labels and confidence scores.

## Member 2 – PLC/External Communication

### Objective
Integrated PLC/external communication using a REST API to simulate sending industrial defect information.

### Implementation
The `src/plc_sender.py` script sends defect information to the FastAPI `/plc/send` endpoint using a JSON payload.

The payload contains:
- x: Defect X-coordinate
- y: Defect Y-coordinate
- defect: Detected defect class
- confidence: Detection confidence score

### Sample Payload

{
  "x": 120,
  "y": 80,
  "defect": "scratches",
  "confidence": 0.87
}

### Testing Result

The PLC sender was tested successfully with the FastAPI server.

HTTP Status: 200

Response:
{
  "status": "success",
  "message": "Defect data received successfully",
  "plc_data": {
    "x": 120.0,
    "y": 80.0,
    "defect": "scratches",
    "confidence": 0.87
  }
}

### Conclusion
PLC/external communication was successfully simulated using REST API and JSON payloads.

## member 3- Prometheus Metrics and Logging

The application includes Prometheus-based monitoring for:
- API request count
- API request latency
- Application uptime
- YOLO inference FPS

Metrics are exposed through the `/metrics` endpoint.

Application logs are stored in:
`logs/app.log`

The application records:
- API request method and endpoint
- HTTP status code
- Request latency
- Prediction completion
- API request method and endpoint
- HTTP status code
- Request latency
- Prediction completion
- Inference time

## Member 4 – Containerization (Docker)

### Objective
Package the entire Real-Time Industrial Defect Detection System into a reproducible Docker container for consistent deployment across environments and edge devices.

### Implementation

**Files Created:**
- `Dockerfile` – Defines the container image with Python 3.10, system dependencies (OpenCV, libGL), and all project requirements.
- `Dockerfile.txt` – Human-readable Docker setup notes and build instructions.
- `docker-compose.yml` – Compose file to orchestrate the FastAPI container service.

**docker-compose.yml Configuration:**
```yaml
services:
  defect-detection-api:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Steps Performed
1. Created `Dockerfile` based on Python 3.10-slim base image.
2. Installed system-level dependencies (libGL, libglib) for OpenCV headless operation.
3. Copied project source files and model weights into the container.
4. Configured `docker-compose.yml` to expose port 8000 and launch the FastAPI server.
5. Verified the compose file structure and service definition.

### Conclusion
The entire system is now containerized and can be deployed on any Docker-compatible machine or edge device with a single `docker-compose up` command.

---

## Member 5 – API Performance Benchmarking & System Validation

### Objective
Perform quantitative performance evaluation of all FastAPI endpoints under real load conditions, identify bottlenecks, and produce a final validation report for the Week 4 system.

### Files Created
- `src/api_benchmark.py` – Automated benchmarking and load testing suite.
- `docs/week4_performance_metrics.txt` – Full benchmark report with latency distributions.
- `requirements.txt` updated – Added `prometheus-client` dependency for metrics support.

### Benchmarking Methodology
- **Tool:** Custom Python benchmarking suite using `concurrent.futures.ThreadPoolExecutor`.
- **Load Levels:** Concurrency 1, 5, and 10 simultaneous clients.
- **Requests per test:** 100 requests per endpoint per concurrency level.
- **Metrics collected:** Success rate, RPS (Requests Per Second), Avg/Min/Max/P50/P90/P99 latency.
- **Endpoints tested:** `GET /health`, `POST /plc/send`, `POST /predict`.

### Results Summary

**GET /health (Availability Check)**

| Concurrency | Success | RPS     | Avg Latency | P99 Latency |
|-------------|---------|---------|-------------|-------------|
| 1           | 100%    | 494.66  | 1.92ms      | 6.02ms      |
| 5           | 100%    | 1118.84 | 4.34ms      | 12.02ms     |
| 10          | 100%    | 1475.25 | 6.53ms      | 21.05ms     |

**POST /plc/send (PLC Communication)**

| Concurrency | Success | RPS     | Avg Latency | P99 Latency |
|-------------|---------|---------|-------------|-------------|
| 1           | 100%    | 433.00  | 2.21ms      | 6.02ms      |
| 5           | 100%    | 955.77  | 5.08ms      | 13.00ms     |
| 10          | 100%    | 1152.01 | 8.40ms      | 28.02ms     |

**POST /predict (YOLO ONNX Inference)**

| Concurrency | Success | RPS   | Avg Latency | P99 Latency |
|-------------|---------|-------|-------------|-------------|
| 1           | 100%    | 14.86 | 67.11ms     | 76.84ms     |
| 5           | 100%    | 15.65 | 312.39ms    | 349.91ms    |
| 10          | 100%    | 15.54 | 625.59ms    | 702.44ms    |

### Analysis & Findings
1. **100% success rate** across all endpoints and all concurrency levels — the system is fully stable under load.
2. **`/health` and `/plc/send`** endpoints are highly scalable, reaching 1,475 RPS and 1,152 RPS respectively at concurrency 10 with P99 latency well under 30ms.
3. **`/predict` (YOLO ONNX)** throughput is CPU-bound at ~15 RPS regardless of concurrency, since ONNX inference runs on a single CPUExecutionProvider thread. For 30+ FPS real-time factory lines, GPU acceleration (TensorRT) is required, as validated in Week 3 at 146.15 FPS.
4. **Single-stream /predict latency:** 67ms average at concurrency=1 is suitable for standard 15 FPS industrial camera inspection pipelines.

### Recommendations
1. Use **TensorRT GPU** runtime in production for maximum throughput (10x over CPU ONNX).
2. Run `uvicorn` with `--workers N` for multi-core CPU parallelism on multi-stream setups.
3. `/health` and `/plc/send` can be scaled horizontally without bottlenecks.

### Conclusion
Week 4 Member 5 task is complete. A comprehensive API benchmarking suite was developed and executed, producing quantitative performance metrics for all system endpoints. The full report is available at `docs/week4_performance_metrics.txt`.

---

## Week 4 – Final System Status

| Member | Person       | Task                                  | Status    |
|--------|--------------|---------------------------------------|-----------|
| 1      | Rukmani Priya| FastAPI Backend & Video Detection     | ✅ Done   |
| 2      | Visalam      | PLC / External REST Communication     | ✅ Done   |
| 3      | Aashmika     | Prometheus Metrics & Logging          | ✅ Done   |
| 4      | Manjesh      | Dockerization & Container Setup       | ✅ Done   |
| 5      | Krishnakumar | API Benchmarking & Performance Report | ✅ Done   |