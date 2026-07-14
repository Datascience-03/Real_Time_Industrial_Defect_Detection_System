# Week 1 Report

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

### Status

Dataset preparation completed successfully and is ready for preprocessing and model training.

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
