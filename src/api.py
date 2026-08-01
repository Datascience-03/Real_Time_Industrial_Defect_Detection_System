from fastapi import FastAPI, UploadFile, File
from PIL import Image
from ultralytics import YOLO
import numpy as np

app = FastAPI(
    title="Industrial Defect Detection API",
    version="1.0"
)

# Load trained model once
MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")

    results = model(image)

    boxes = results[0].boxes

    if len(boxes) == 0:
        return {
            "filename": file.filename,
            "message": "No defect detected",
            "detections": 0
        }

    best_box = boxes[0]

    class_id = int(best_box.cls.item())
    confidence = float(best_box.conf.item())

    class_name = model.names[class_id]

    return {
        "filename": file.filename,
        "predicted_class": class_name,
        "confidence": round(confidence, 4),
        "detections": len(boxes),
        "message": "Inference completed successfully"
    }