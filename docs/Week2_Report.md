# Week 2 Report

## Project Title
Real-Time Industrial Defect Detection System
member 1
## Week 2 Objectives
- Prepared the dataset for YOLOv8 training.
- Organized images and labels into train, validation, and test folders.
- Configured the `data.yaml` file with six defect classes.
- Installed and verified the Ultralytics YOLOv8 framework.
- Trained the YOLOv8n object detection model for 20 epochs.
- Evaluated the model using validation metrics.

## Work Completed
- Verified dataset structure and annotations.
- Configured training parameters (640 image size, batch size 16, 20 epochs).
- Successfully trained the model using YOLOv8n.
- Generated training outputs including:
  - Best and last model weights (`best.pt`, `last.pt`)
  - Training graphs
  - Confusion matrix
  - Labels visualization
  - Training logs (`results.csv`)
- Achieved excellent detection performance with approximately **99.5% mAP@50**.

## Challenges Faced
- Corrected the dataset folder structure.
- Fixed `data.yaml` configuration issues.
- Resolved training output directory organization.

## Outcome
The YOLOv8 model was successfully trained and validated. The generated model weights (`best.pt`) will be used in the next phase for real-time defect detection and performance testing.