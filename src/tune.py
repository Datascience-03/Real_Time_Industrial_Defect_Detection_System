from ultralytics import YOLO
from pathlib import Path

# ===========================
# Project Root Directory
# ===========================
ROOT = Path(__file__).resolve().parent.parent

# ===========================
# File Paths
# ===========================
MODEL_PATH = ROOT / "yolov8n.pt"
DATA_PATH = ROOT / "dataset" / "data.yaml"
PROJECT_DIR = ROOT / "runs" / "detect"

# ===========================
# Training Hyperparameters
# ===========================
EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SIZE = 640
LEARNING_RATE = 0.01
EXPERIMENT_NAME = "baseline_tuning"


def load_model():
    """
    Load the pretrained YOLOv8 model.
    """
    return YOLO(str(MODEL_PATH))


def train_model(model):
    """
    Train the YOLO model using configurable hyperparameters.
    """

    results = model.train(
        data=str(DATA_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        project=str(PROJECT_DIR),
        name=EXPERIMENT_NAME,
        exist_ok=True,
    )

    return results


def main():
    model = load_model()
    train_model(model)


if __name__ == "__main__":
    main()