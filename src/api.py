from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import time

app = FastAPI(
    title="Industrial Defect Detection API",
    description="""
REST API for Industrial Surface Defect Detection using YOLOv8.

Features
--------
• Image Upload
• Defect Prediction
• Health Monitoring
• Version Information
• Swagger Documentation
""",
    version="1.1.0",
    contact={
        "name": "Team Datascience-03"
    }
)

# ----------------------------------------------------
# Load YOLO Model
# ----------------------------------------------------

MODEL_PATH = "runs/detect/train/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
    model_loaded = True
except Exception:
    model_loaded = False


# ----------------------------------------------------
# Home
# ----------------------------------------------------

@app.get(
    "/",
    tags=["General"],
    summary="Home",
    description="Returns welcome message."
)
def home():

    return {
        "message": "Industrial Defect Detection API is Running!"
    }


# ----------------------------------------------------
# Health Check
# ----------------------------------------------------

@app.get(
    "/health",
    tags=["General"],
    summary="Check API Health",
    description="Returns current API status and confirms whether the YOLO model is loaded."
)
def health():

    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


# ----------------------------------------------------
# Version
# ----------------------------------------------------

@app.get(
    "/version",
    tags=["General"],
    summary="API Version",
    description="Returns API version information."
)
def version():

    return {
        "application": "Industrial Defect Detection API",
        "version": "1.1.0",
        "framework": "FastAPI",
        "model": "YOLOv8",
        "status": "Stable"
    }


# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Industrial Defect",
    description="Uploads an image and predicts the defect using the trained YOLOv8 model."
)
async def predict(file: UploadFile = File(...)):

    if not model_loaded:
        raise HTTPException(
            status_code=500,
            detail="YOLO model not loaded."
        )

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    width, height = image.size

    start = time.time()

    results = model(image)

    end = time.time()

    inference_time = round((end - start) * 1000, 2)

    boxes = results[0].boxes

    if len(boxes) == 0:

        return {
            "filename": file.filename,
            "model": "YOLOv8",
            "detections": 0,
            "width": width,
            "height": height,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inference_time_ms": inference_time,
            "message": "No defect detected"
        }

    best_box = boxes[0]

    class_id = int(best_box.cls.item())

    confidence = float(best_box.conf.item())

    class_name = model.names[class_id]

    return {

        "filename": file.filename,

        "model": "YOLOv8",

        "predicted_class": class_name,

        "confidence": round(confidence, 4),

        "detections": len(boxes),

        "width": width,

        "height": height,

        "inference_time_ms": inference_time,

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "message": "Prediction completed successfully"

    }