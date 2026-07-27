from ultralytics import YOLO

print("Loading ONNX model...")

model = YOLO("runs/detect/train/weights/best.onnx")

print("Running inference...")

results = model.predict(
    source="dataset/test/images",
    save=True,
    imgsz=640
)

print("ONNX model inference completed successfully!")