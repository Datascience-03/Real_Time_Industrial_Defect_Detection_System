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
# ===========================
# Hyperparameter Experiments
# ===========================

EXPERIMENTS = [
    {
        "name": "baseline_tuning",
        "epochs": 20,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01
    }
]

def load_model():
    """
    Load the pretrained YOLOv8 model.
    """
    return YOLO(str(MODEL_PATH))


def train_model(model, data_path, experiment):
    """
    Train YOLO using a specified hyperparameter configuration.
    """

    results = model.train(
        data=str(data_path),
        epochs=experiment["epochs"],
        imgsz=experiment["imgsz"],
        batch=experiment["batch"],
        lr0=experiment["lr0"],
        project=str(PROJECT_DIR),
        name=experiment["name"],
        exist_ok=True
    )

    return results

def main():
    model = load_model()

    for experiment in EXPERIMENTS:
        print(f"\nRunning Experiment: {experiment['name']}")

        train_model(
            model=model,
            data_path=DATA_PATH,
            experiment=experiment
        )


if __name__ == "__main__":
    main()