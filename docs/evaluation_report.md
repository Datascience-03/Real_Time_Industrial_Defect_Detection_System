# Model Evaluation Report
**Project:** Real-Time Industrial Defect Detection System  
**Task:** Member 4 - Model Evaluation  
**Date:** July 16, 2026  

---

## 1. Executive Summary
This report presents the performance evaluation of the Defect Detection System. The evaluation was carried out using synthetic validation data representing common industrial defects: **crazing**, **patches**, and **scratches**. The model's predictions were compared against ground truth bounding boxes to evaluate detection accuracy using Mean Average Precision (mAP) at an Intersection over Union (IoU) threshold of 0.5.

The model achieved an overall **mAP@0.5 of 0.3889 (38.89%)**. While the system demonstrates capability in detecting linear defects such as *scratches*, limitations were observed in detecting localized surface variations such as *patches*.

---

## 2. Quantitative Evaluation Metrics
The quantitative results calculated from the validation dataset are summarized in the table below:

| Class Name | Ground Truths (GT) | Predictions (Preds) | Average Precision (AP @0.5) | Detection Status |
| :--- | :---: | :---: | :---: | :---: |
| **crazing** | 2 | 2 | 0.5000 (50.00%) | Moderate Detection |
| **patches** | 1 | 0 | 0.0000 (0.00%) | No Detection (Failed) |
| **scratches** | 3 | 4 | 0.6667 (66.67%) | High Detection |
| **Overall System** | **6** | **6** | **0.3889 (38.89%)** | Tuning Required |

---

## 3. Analysis of Defect Categories

### A. Scratches (AP: 66.67%)
* **Performance:** This class yielded the highest performance.
* **Analysis:** With 3 ground truth instances, the model produced 4 predictions, indicating a high recall rate but introducing a risk of false positives. The high AP suggests that the spatial features of scratches are distinct and reasonably learned by the current configuration.

### B. Crazing (AP: 50.00%)
* **Performance:** Moderate performance.
* **Analysis:** The model matched the count of instances (2 GTs, 2 Predictions) exactly. However, the spatial alignment (IoU) of these predictions limited the final AP to 50.00%.

### C. Patches (AP: 0.00%)
* **Performance:** Critical failure.
* **Analysis:** The model failed to predict the single patch instance. This suggests a lack of robust feature representation for patch defects, which typically possess less defined boundaries compared to scratches.

---

## 4. Performance Plots
The class-wise Precision-Recall (PR) curves have been generated and saved locally to evaluate the trade-offs at different confidence thresholds:

* **Crazing PR Curve:** `outputs/reports/crazing_pr_curve.png`
* **Patches PR Curve:** `outputs/reports/patches_pr_curve.png`
* **Scratches PR Curve:** `outputs/reports/scratches_pr_curve.png`

*(Note: Please refer to the `outputs/reports/` folder to view the high-resolution visualization plots)*

---

## 5. Recommendations for Next Stages (Hand-off)
1. **To Member 3 (Hyperparameter Tuning):** Focus tuning efforts on adjusting the anchor box sizes and training epochs. The poor performance on `patches` indicates the model needs optimized hyperparameter configurations (e.g., learning rate adjustments or custom anchors) to detect diverse aspect ratios.
2. **To Member 5 (Error Analysis):** Focus manual error analysis on the `patches` and `crazing` instances to identify why bounding box localization failed.