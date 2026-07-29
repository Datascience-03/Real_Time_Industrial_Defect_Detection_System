import cv2
import time
from ultralytics import YOLO

# 1. Load the model (Works with 'model.engine', 'model.onnx', or 'yolov8n.pt')
model = YOLO("model.engine")

# 2. Open Video Stream (Use 0 for webcam, or 'video.mp4' for a video file)
cap = cv2.VideoCapture(0)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video stream ended or camera not found.")
        break

    # 3. Perform Inference
    results = model(frame)

    # 4. Draw bounding boxes on frame
    annotated_frame = results[0].plot()

    # 5. Calculate FPS
    curr_time = time.time()
    time_diff = curr_time - prev_time
    fps = 1.0 / time_diff if time_diff > 0 else 0
    prev_time = curr_time

    # 6. Display FPS Counter on top-left of video
    cv2.putText(
        annotated_frame,
        f"TensorRT FPS: {fps:.2f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 7. Display Output
    cv2.imshow("Real-Time Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()