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

### member manjesh progress

Dataset preparation completed successfully and is ready for preprocessing and model training.
# Week 1 - Preprocessing Pipeline Integrated

### Accomplishments:
- **Preprocessing Pipeline (Member 4):** Completed `preprocess.py`.
- **Functionality:** Implemented automated resizing (640x640) and normalization (0-1).
- **Integration:** Successfully linked the `raw` dataset folders to the `augmented` destination. The pipeline now automatically processes 350+ images across all defect categories (crazing, scratches, etc.).
- **Status:** Integrated, verified, and ready for training.