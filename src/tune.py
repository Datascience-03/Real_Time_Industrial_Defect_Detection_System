from ultralytics import YOLO
from pathlib import Path
import csv

# ==================================================
# Project Paths
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "yolov8n.pt"
DATA_PATH = ROOT / "dataset" / "data.yaml"

PROJECT_DIR = ROOT / "runs" / "detect"

# ==================================================
# Hyperparameter Experiments
# ==================================================

EXPERIMENTS = [

    {
        "name": "baseline",
        "epochs": 20,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.01
    },

    {
        "name": "batch16",
        "epochs": 20,
        "batch": 16,
        "imgsz": 640,
        "lr0": 0.01
    },

    {
        "name": "lr005",
        "epochs": 20,
        "batch": 8,
        "imgsz": 640,
        "lr0": 0.005
    }

]


def load_model():
    return YOLO(str(MODEL_PATH))


def train_model(model, experiment):

    results = model.train(

        data=str(DATA_PATH),

        epochs=experiment["epochs"],

        batch=experiment["batch"],

        imgsz=experiment["imgsz"],

        lr0=experiment["lr0"],

        project=str(PROJECT_DIR),

        name=experiment["name"],

        exist_ok=True

    )

    return results


def save_summary(summary):

    output = ROOT / "docs" / "hyperparameter_results.csv"

    with open(output, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Experiment",
                "Epochs",
                "Batch",
                "Image Size",
                "Learning Rate"
            ]
        )

        writer.writerows(summary)

    print(f"\nResults saved to {output}")


def main():

    summary = []

    for experiment in EXPERIMENTS:

        print("=" * 50)
        print(f"Running {experiment['name']}")
        print("=" * 50)

        model = load_model()

        train_model(model, experiment)

        summary.append([
            experiment["name"],
            experiment["epochs"],
            experiment["batch"],
            experiment["imgsz"],
            experiment["lr0"]
        ])

    save_summary(summary)


if __name__ == "__main__":
    main()