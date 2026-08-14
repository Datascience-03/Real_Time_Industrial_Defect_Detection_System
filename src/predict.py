from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
from src.utils import get_checkpoint_path

ck = get_checkpoint_path()
model = YOLO(ck if ck is not None else "yolov8n.pt")

# Input folder containing test images
source_folder = "dataset/test/images"

# Run prediction
results = model.predict(
    source=source_folder,
    save=True,
    conf=0.25
)

print("\nPrediction completed successfully!")
print("Results saved in: runs/detect/predict")