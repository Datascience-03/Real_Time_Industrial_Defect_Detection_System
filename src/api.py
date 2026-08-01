from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
from datetime import datetime
import time

app = FastAPI(
    title="Industrial Defect Detection API",
    description="API for detecting industrial surface defects using YOLOv8",
    version="1.0"
)

# Load trained YOLO model
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is Running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate uploaded image
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    width, height = image.size

    # Start inference timer
    start = time.time()

    results = model(image)

    end = time.time()

    inference_time = round((end - start) * 1000, 2)

    boxes = results[0].boxes

    # No detections
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

    # Highest confidence prediction
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