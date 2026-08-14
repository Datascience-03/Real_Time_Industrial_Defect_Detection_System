from ultralytics import YOLO
import os

# Use `model.onnx` if available, otherwise fallback to `yolov8n.onnx` in repo root
# Prefer the exported trained ONNX if present
possible_paths = [
    "runs/detect/train/weights/best.onnx",
    "model.onnx",
    "yolov8n.onnx",
]

model_path = next((p for p in possible_paths if os.path.exists(p)), possible_paths[-1])
print(f"Loading ONNX model from: {model_path}")
model = YOLO(model_path, task="detect")

image_path = "dataset/test/images/rolled-in_scale_277.jpg"

results = model.predict(
    source=image_path,
    conf=0.05,
    iou=0.45,
    imgsz=640,
    verbose=True
)

for result in results:
    print("\n===== ONNX DETECTIONS =====")

    if result.boxes is None or len(result.boxes) == 0:
        print("NO DETECTIONS")
    else:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(
                f"Class ID: {cls_id}, "
                f"Class Name: {model.names[cls_id]}, "
                f"Confidence: {confidence:.4f}"
            )

    result.save(filename="onnx_test_result.jpg")