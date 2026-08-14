from ultralytics import YOLO

print("Loading model...")

from src.utils import get_checkpoint_path

ck = get_checkpoint_path()
model = YOLO(ck if ck is not None else "yolov8n.pt")

print("Exporting model to ONNX...")

model.export(format="onnx")

print("ONNX model exported successfully!")