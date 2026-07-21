from ultralytics import YOLO
import os

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

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