 Week 1 Report

## Member rukmani Progress

### Tasks Completed

- Downloaded the NEU Surface Defect Dataset.
- Organized the dataset into six defect classes.
- Split the dataset into:
  - Training (80%)
  - Validation (10%)
  - Testing (10%)
- Verified the dataset distribution.
- Checked all images for corruption.
- No corrupted images were found.

### Dataset Classes

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled In Scale
- Scratches

### member manjesh progress

Dataset preparation completed successfully and is ready for preprocessing and model training.
<<<<<<< HEAD
# Week 1 - Preprocessing Pipeline Integrated

### Accomplishments:
- **Preprocessing Pipeline (Member 4):** Completed `preprocess.py`.
- **Functionality:** Implemented automated resizing (640x640) and normalization (0-1).
- **Integration:** Successfully linked the `raw` dataset folders to the `augmented` destination. The pipeline now automatically processes 350+ images across all defect categories (crazing, scratches, etc.).
- **Status:** Integrated, verified, and ready for training.
=======

# Week 1 Visalam report 
# YOLO Annotation Notes

This dataset (NEU steel surface defect dataset) originally had **no bounding box
ground truth** — it's a classification dataset organized as one folder per defect
class (crazing, inclusion, patches, pitted_surface, rolled_in_scale, scratches),
with 200x200 images already cropped/centered on a single defect each.

Since there is no true localization data available, bounding boxes in this
package were generated using the standard convention for repurposing this
dataset for YOLO detection: **one box per image, centered, covering 90% of the
image dimensions**, labeled with the class from its original folder.

Label format (YOLO): `class_id x_center y_center width height` (all normalized 0-1)

⚠️ These are NOT ground-truth localized boxes — they mark "this image contains
this defect somewhere," not the defect's exact position. If you need accurate
per-defect localization for a real detection model, you'll need to either:
- manually annotate a subset with a tool like LabelImg / CVAT / Roboflow, or
- use a dataset that already has bounding box annotations (e.g. NEU-DET, which
  is the detection variant of this dataset with real box coordinates).

Also fixed a typo in the original data.yaml: "scratchess" → "scratches".
>>>>>>> e4e592cffce95ef47616b10d47b1e48c404ef4a6

# Data manjesh progress Augmentation Module (`augmentation.py`)

This module contains the core data augmentation pipeline for the **Real-Time Industrial Defect Detection System**. It uses the `albumentations` library to expand the dataset size by introducing variations in spatial orientation, lighting conditions, and sensor noise, ensuring robust training for object detection models.

---

## ⚙️ Functionality Overview

The script processes raw industrial images and their corresponding YOLO-formatted bounding box text files simultaneously, applying coordinate-safe transformations so the labels always line up perfectly with the mutated images.

### Transformations Applied:
* **Horizontal Flip:** Flips the image horizontally with a 50% probability.
* **Random Rotate 90:** Rotates the image randomly by 90, 180, or 270 degrees to simulate varying camera mounting angles.
* **Random Brightness & Contrast:** Shifts brightness and contrast by $\pm 20\%$ to replicate fluctuating factory floor illumination.
* **Gaussian Noise:** Injects random noise (variance range: 10.0 to 50.0) to emulate camera sensor grain and transmission interference.

---

## 🛠️ Usage & Configuration

### 1. Prerequisites
Make sure the necessary computer vision and augmentation packages are installed:
```bash
pip install opencv-python albumentations

