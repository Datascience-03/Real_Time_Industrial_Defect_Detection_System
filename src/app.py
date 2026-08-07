from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
import time


# ==================================================
# Project Root
# ==================================================

ROOT = Path(__file__).resolve().parent.parent


# ==================================================
# Model Path
# ==================================================

MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.onnx"
)


# ==================================================
# Load YOLO ONNX Model
# ==================================================

print("Loading YOLO ONNX model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
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

app = FastAPI(
    title="Real-Time Industrial Defect Detection API",
    description="FastAPI backend for industrial defect detection using YOLO ONNX.",
    version="1.0.0"
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
async def predict(file: UploadFile = File(...)):

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
        conf=0.25,
        verbose=False
    )

    inference_time = time.time() - start_time


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

