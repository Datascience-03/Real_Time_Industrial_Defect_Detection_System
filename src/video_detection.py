import cv2
from pathlib import Path
from ultralytics import YOLO

# ==================================================
# Project Root
# ==================================================
ROOT = Path(__file__).resolve().parent.parent

# ==================================================
# Model Path
# ==================================================
MODEL_PATH = (
    ROOT
    / "runs"
    / "detect"
    / "train"
    / "weights"
    / "best.pt"
)

# ==================================================
# Video Folder
# ==================================================
VIDEO_DIR = ROOT / "dataset" / "vedios"

# ==================================================
# Output Folder
# ==================================================
OUTPUT_DIR = ROOT / "outputs" / "video_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Settings
# ==================================================
IMAGE_SIZE = 640
CONFIDENCE = 0.25


def process_video(model, video_path):

    print("\n========================================")
    print(f"Processing: {video_path.name}")
    print("========================================")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    # Video information
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(f"Video FPS    : {fps}")
    print(f"Video Width  : {width}")
    print(f"Video Height : {height}")
    print(f"Total Frames : {total_frames}")

    # Output filename automatically matches input
    output_video = (
        OUTPUT_DIR
        / f"{video_path.stem}_detected.mp4"
    )

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        print("ERROR: Cannot create output video.")
        return

    frame_count = 0
    detection_count = 0

    print("Video detection started...")
    print("The video will be processed without displaying a window.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # YOLO prediction
        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]

        # Count detections
        if result.boxes is not None:
            detection_count += len(result.boxes)

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        # Save processed frame
        writer.write(annotated_frame)

        frame_count += 1

        # Show progress
        if frame_count % 100 == 0:
            print(
                f"Processed frames: "
                f"{frame_count}/{total_frames}"
            )

    # Release resources
    cap.release()
    writer.release()

    print("----------------------------------------")
    print(f"Completed: {video_path.name}")
    print(f"Processed frames : {frame_count}")
    print(f"Total detections : {detection_count}")
    print(f"Output video     : {output_video}")
    print("----------------------------------------")


def main():

    # Check model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Check video folder
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video folder not found: {VIDEO_DIR}"
        )

    # Load YOLO model
    print("Loading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("Classes:", model.names)

    # Find all MP4 videos
    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    if not videos:
        raise FileNotFoundError(
            f"No MP4 videos found in {VIDEO_DIR}"
        )

    print("\nVideos found:")

    for video in videos:
        print(f" - {video.name}")

    # Process every video
    for video_path in videos:
        process_video(
            model,
            video_path
        )

    print("\n========================================")
    print("ALL VIDEOS PROCESSED SUCCESSFULLY")
    print("========================================")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()