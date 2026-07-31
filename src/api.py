from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Week 4 FastAPI Backend",
    version="1.0.0"
)

# =====================================================
# HOME
# =====================================================
@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running."
    }


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Industrial Defect Detection",
        "version": "1.0.0"
    }


# =====================================================
# IMAGE PREPROCESSING
# =====================================================
def preprocess_image(image: Image.Image):
    """
    Resize image and normalize pixel values.
    """

    image = image.resize((640, 640))
    image_array = np.array(image).astype(np.float32) / 255.0

    return image_array


# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and perform preprocessing.
    Actual YOLO inference will be integrated later.
    """

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed_image = preprocess_image(image)

    height, width = processed_image.shape[:2]

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "message": "Image received and preprocessed successfully.",
        "predicted_class": "No Defect",
        "confidence": 0.98
    }