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
- Inference time