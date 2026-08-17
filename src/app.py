from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse, StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import cv2
import time
import logging
import os
import tempfile
import uuid

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .metrics import (
    record_request,
    record_inference_fps,
    update_uptime
)
# ==================================================
# Application Logging
# ==================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

logger.info("Application logging initialized.")
# ==================================================
# Project Root
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

# ==================================================
# Model Path
# ==================================================

MODEL_CANDIDATES = (
    ROOT / "runs" / "detect" / "train" / "weights" / "best.onnx",
    ROOT / "release" / "best.onnx",
    ROOT / "model.onnx",
)
MODEL_PATH = next((path for path in MODEL_CANDIDATES if path.exists()), None)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024

# ==================================================
# Video Configuration
# ==================================================

MAX_VIDEO_SIZE_BYTES = 200 * 1024 * 1024

VIDEO_UPLOAD_DIR = ROOT / "uploads" / "videos"
VIDEO_OUTPUT_DIR = ROOT / "outputs" / "videos"

VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/mov",
    "video/quicktime",
}

# ==================================================
# Load YOLO ONNX Model
# ==================================================

print("Loading YOLO ONNX model...")

if MODEL_PATH is None:
    raise FileNotFoundError(
        "ONNX model not found. Checked: "
        + ", ".join(str(path) for path in MODEL_CANDIDATES)
    )

print(f"Model path: {MODEL_PATH}")

model = YOLO(
    str(MODEL_PATH),
    task="detect"
)

print("YOLO ONNX model loaded successfully.")
print("Model classes:", model.names)

# ==================================================
# FastAPI Application
# ==================================================

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the inference backend before accepting user uploads."""
    warmup_image = np.zeros((640, 640, 3), dtype=np.uint8)
    start_time = time.perf_counter()
    model.predict(source=warmup_image, imgsz=640, conf=0.25, verbose=False)
    logger.info("Model warm-up completed in %.3fs", time.perf_counter() - start_time)
    yield


app = FastAPI(
    title="Real-Time Industrial Defect Detection API",
    description="FastAPI backend for industrial defect detection using YOLO ONNX.",
    version="1.0.0",
    lifespan=lifespan,
)

# ==================================================
# Serve Static Dashboard
# ==================================================

STATIC_DIR = ROOT / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))

# ==================================================
# Prometheus Request Monitoring Middleware
# ==================================================

@app.middleware("http")
async def metrics_middleware(request, call_next):

    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time

    record_request(
        method=request.method,
        endpoint=request.url.path,
        latency=latency
    )

    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Latency: {latency:.4f}s"
    )

    return response
# ==================================================
# Prometheus Metrics Endpoint
# ==================================================

@app.get("/metrics")
def metrics():

    update_uptime()

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ==================================================
# Health Check Endpoint
# ==================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "loaded",
        "model_path": str(MODEL_PATH),
        "classes": model.names
    }


# ==================================================
# Prediction Endpoint
# ==================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Minimum detection confidence"),
):

    # --------------------------------------------------
    # Check file type
    # --------------------------------------------------

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, or PNG image."
        )

    # --------------------------------------------------
    # Read uploaded file
    # --------------------------------------------------

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded image exceeds the 20 MB size limit."
        )

    # --------------------------------------------------
    # Convert bytes to OpenCV image
    # --------------------------------------------------

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode the uploaded image."
        )

    # --------------------------------------------------
    # Run YOLO prediction
    # --------------------------------------------------

    start_time = time.time()

    results = model.predict(
        source=image,
        imgsz=640,
        conf=conf,
        verbose=False
    )

    inference_time = time.time() - start_time
    logger.info(
        "Prediction completed: %s | Inference time: %.4fs | conf=%.2f",
        file.filename,
        inference_time,
        conf,
    )

    # --------------------------------------------------
    # Calculate inference FPS
    # --------------------------------------------------

    inference_fps = (
        1.0 / inference_time
        if inference_time > 0
        else 0
    )

    record_inference_fps(inference_fps)

    # --------------------------------------------------
    # Extract predictions
    # --------------------------------------------------

    detections = []

    result = results[0]

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            # Class ID
            class_id = int(
                boxes.cls[i].item()
            )

            # Confidence
            confidence = float(
                boxes.conf[i].item()
            )

            # Bounding box coordinates
            coordinates = boxes.xyxy[i].tolist()

            # Class name
            class_name = model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(
                    confidence,
                    4
                ),
                "bbox": [
                    round(float(coordinates[0]), 2),
                    round(float(coordinates[1]), 2),
                    round(float(coordinates[2]), 2),
                    round(float(coordinates[3]), 2)
                ]
            })

    # --------------------------------------------------
    # Return API response
    # --------------------------------------------------

    return {
        "filename": file.filename,
        "detections": detections,
        "detection_count": len(detections),
        "inference_time_ms": round(
            inference_time * 1000,
            2
        )
    }


# ==================================================
# PLC / External Communication
# ==================================================

class PLCDefectPayload(BaseModel):

    x: float
    y: float
    defect: str
    confidence: float


@app.post("/plc/send")
def receive_plc_data(payload: PLCDefectPayload):

    return {
        "status": "success",
        "message": "Defect data received successfully",
        "plc_data": {
            "x": payload.x,
            "y": payload.y,
            "defect": payload.defect,
            "confidence": payload.confidence
        }
    }

    # ==================================================
# Video Prediction Endpoint
# ==================================================

@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Query(
        0.25,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence"
    ),
):
    """
    Upload an MP4 video, run YOLO detection on every frame,
    and return the annotated video.
    """

    # --------------------------------------------------
    # Check video type
    # --------------------------------------------------

    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a video file. "
                "Supported formats: MP4, AVI, MOV."
            )
        )

    # --------------------------------------------------
    # Read uploaded video
    # --------------------------------------------------

    video_bytes = await file.read()

    if not video_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded video is empty."
        )

    if len(video_bytes) > MAX_VIDEO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Uploaded video exceeds the 200 MB size limit."
        )

    # --------------------------------------------------
    # Create unique filenames
    # --------------------------------------------------

    video_id = uuid.uuid4().hex

    input_path = VIDEO_UPLOAD_DIR / f"{video_id}_input.mp4"
    output_path = VIDEO_OUTPUT_DIR / f"{video_id}_detected.mp4"

    # --------------------------------------------------
    # Save uploaded video
    # --------------------------------------------------

    try:
        with open(input_path, "wb") as video_file:
            video_file.write(video_bytes)

        # --------------------------------------------------
        # Open video
        # --------------------------------------------------

        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Unable to open the uploaded video."
            )

        # --------------------------------------------------
        # Get video properties
        # --------------------------------------------------

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # --------------------------------------------------
        # Create output video writer
        # --------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()

            raise HTTPException(
                status_code=500,
                detail="Unable to create output video."
            )

        # --------------------------------------------------
        # Process every frame
        # --------------------------------------------------

        frame_number = 0
        total_detections = 0

        start_time = time.time()

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # --------------------------------------------------
            # YOLO inference
            # --------------------------------------------------

            results = model.predict(
                source=frame,
                imgsz=640,
                conf=conf,
                verbose=False
            )

            result = results[0]

            frame_detections = 0

            # --------------------------------------------------
            # Draw detections
            # --------------------------------------------------

            if result.boxes is not None:

                boxes = result.boxes

                for i in range(len(boxes)):

                    class_id = int(
                        boxes.cls[i].item()
                    )

                    confidence = float(
                        boxes.conf[i].item()
                    )

                    coordinates = boxes.xyxy[i].tolist()

                    x1 = int(coordinates[0])
                    y1 = int(coordinates[1])
                    x2 = int(coordinates[2])
                    y2 = int(coordinates[3])

                    class_name = model.names[class_id]

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Detection label
                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    frame_detections += 1

            total_detections += frame_detections

            # --------------------------------------------------
            # Add frame information
            # --------------------------------------------------

            cv2.putText(
                frame,
                f"Frame: {frame_number}/{frame_count}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Detections: {frame_detections}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # --------------------------------------------------
            # Write processed frame
            # --------------------------------------------------

            writer.write(frame)

        # --------------------------------------------------
        # Release resources
        # --------------------------------------------------

        cap.release()
        writer.release()

        processing_time = time.time() - start_time

        logger.info(
            "Video prediction completed: %s | "
            "Frames=%d | Detections=%d | Time=%.2fs",
            file.filename,
            frame_number,
            total_detections,
            processing_time
        )

        # --------------------------------------------------
        # Check output
        # --------------------------------------------------

        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Output video was not created."
            )

        # --------------------------------------------------
        # Return processed video
        # --------------------------------------------------

        return FileResponse(
            path=str(output_path),
            media_type="video/mp4",
            filename=f"detected_{file.filename}"
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Video processing failed: %s",
            str(e)
        )

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

    finally:

        # --------------------------------------------------
        # Remove temporary input video
        # --------------------------------------------------

        try:
            if input_path.exists():
                input_path.unlink()
        except Exception:
            pass
