import cv2
from pathlib import Path

# ==================================================
# Project Root
# ==================================================
ROOT = Path(__file__).resolve().parent.parent

# ==================================================
# Video File
# ==================================================
VIDEO_PATH = ROOT / "dataset" / "vedios" / "sample1.mp4"

# ==================================================
# Output Folder
# ==================================================
OUTPUT_DIR = ROOT / "outputs" / "video_frames"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# Frame Settings
# ==================================================
FRAME_WIDTH = 640
FRAME_HEIGHT = 640

SAVE_FRAMES = False      # Change to True if you want to save frames
DISPLAY_WINDOW = True


def open_video_source(video_path):
    """
    Open a video file.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    return cap


def preprocess_frame(frame):
    """
    Resize frame for YOLO.
    """
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    return frame


def save_frame(frame, frame_number):
    """
    Save processed frame.
    """
    filename = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"
    cv2.imwrite(str(filename), frame)


def main():

    cap = open_video_source(VIDEO_PATH)

    frame_count = 0

    print("Video Started...")
    print("Press 'q' to Quit")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video Finished.")
            break

        frame = preprocess_frame(frame)

        if SAVE_FRAMES:
            save_frame(frame, frame_count)

        if DISPLAY_WINDOW:
            cv2.imshow("Industrial Video Capture", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Processed Frames : {frame_count}")


if __name__ == "__main__":
    main()