from ultralytics import YOLO
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent


def load_model(model_path=ROOT / "yolov8n.pt"):
    model = YOLO(str(model_path))
    return model


def train_model(model, data_path):
    results = model.train(
        data=str(data_path),
        epochs=20,
        imgsz=640,
        batch=8,
        project=str(ROOT / "runs" / "detect"),
        name="baseline_tuning",
        exist_ok=True
    )
    return results


def main():
    model = load_model()

    train_model(
        model=model,
        data_path=ROOT / "dataset" / "data.yaml"
    )


if __name__ == "__main__":
    main()