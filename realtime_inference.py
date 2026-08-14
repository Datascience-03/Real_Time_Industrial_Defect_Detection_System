import argparse
import os
import time
import cv2
from ultralytics import YOLO


def choose_model_path():
    candidates = [
        "runs/detect/train/weights/best.onnx",
        "model.onnx",
        "yolov8n.onnx",
        "yolov8n.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


def run_video(model, cap, show=True):
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream ended or camera not found.")
            break

        results = model(frame)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        time_diff = curr_time - prev_time
        fps = 1.0 / time_diff if time_diff > 0 else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if show:
            cv2.imshow("Real-Time Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # when not showing, just write last frame to disk for inspection
            cv2.imwrite("realtime_last_frame.jpg", annotated_frame)
            break

    cap.release()
    if show:
        cv2.destroyAllWindows()


def run_image(model, image_path, show=True):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return
    results = model(img)
    annotated = results[0].plot()
    out_path = os.path.basename(image_path).rsplit('.', 1)[0] + "_annotated.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated image to {out_path}")
    if show:
        cv2.imshow("Image Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", default="dataset/vedios/sample1.mp4", help="Video file, camera index, or image path")
    parser.add_argument("--model", "-m", default=None, help="Path to ONNX/PT/engine model file")
    parser.add_argument("--conf", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Do not show GUI windows (save outputs instead)")
    args = parser.parse_args()

    model_path = args.model if args.model else choose_model_path()
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    source = args.source
    show = not args.no_display

    # If source is integer-like, treat as camera index
    try:
        cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
        run_video(model, cap, show=show)
        return
    except Exception:
        pass

    # If source is a file
    if os.path.exists(source):
        # check image vs video by extension
        ext = os.path.splitext(source)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            run_image(model, source, show=show)
        else:
            cap = cv2.VideoCapture(source)
            run_video(model, cap, show=show)
    else:
        print(f"[WARN] Source '{source}' not found. Falling back to webcam index 0.")
        cap = cv2.VideoCapture(0)
        run_video(model, cap, show=show)


if __name__ == "__main__":
    main()