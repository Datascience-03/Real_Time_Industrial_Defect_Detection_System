from ultralytics import YOLO

print("Loading model...")

model = YOLO("runs/detect/train/weights/best.pt")

print("Exporting model to ONNX...")

model.export(format="onnx")

print("ONNX model exported successfully!")