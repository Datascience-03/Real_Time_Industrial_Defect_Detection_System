import cv2
import time
import os
from ultralytics import YOLO

def main():
    # 1. Check and Load Model (.engine / .onnx / .pt)
    model_path = "yolov8n.engine"
    
    if not os.path.exists(model_path):
        print(f"Warning: '{model_path}' not found. Falling back to 'yolov8n.pt'.")
        model_path = "yolov8n.pt"

    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # 2. Initialize OpenCV Video Capture
    # Use 0 for webcam or replace with path to a video file: cap = cv2.VideoCapture('sample.mp4')
    cap = cv2.VideoCapture(0)

    # Set frame dimensions for optimized performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0
    fps = 0

    print("\nStarting Real-Time Inference... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or camera unavailable.")
            break

        # Calculate FPS timing
        curr_time = time.time()

        # 3. Perform Inference
        results = model(frame, stream=True, verbose=False)

        # 4. Render Bounding Boxes and Labels
        for r in results:
            annotated_frame = r.plot()

        # Calculate FPS
        time_diff = curr_time - prev_time
        if time_diff > 0:
            fps = 1.0 / time_diff
        prev_time = curr_time

        # 5. Overlay FPS Counter on Frame
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            annotated_frame,
            fps_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),  # Green color
            2,
            cv2.LINE_AA
        )

        # Display window
        cv2.imshow("Real-Time TensorRT Object Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    