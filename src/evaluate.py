import os
import numpy as np
import matplotlib.pyplot as plt
import copy

# Ensure the output directories exist
REPORTS_DIR = "outputs/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def calculate_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1: list or tuple of [xmin, ymin, xmax, ymax]
    - box2: list or tuple of [xmin, ymin, xmax, ymax]

    Returns:
    - Float IoU value between 0.0 and 1.0
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # Calculate union area
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area_box1 + area_box2 - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def calculate_ap(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculates Average Precision (AP) for a single class using the 11-point/all-point interpolation.

    Parameters:
    - gt_boxes: list of dicts: {'image_id': str/int, 'box': [xmin, ymin, xmax, ymax]}
    - pred_boxes: list of dicts: {'image_id': str/int, 'confidence': float, 'box': [xmin, ymin, xmax, ymax]}
    - iou_threshold: float, overlap threshold (default 0.5)

    Returns:
    - ap: float, average precision value
    - precisions: list/array of precision values
    - recalls: list/array of recall values
    """
    total_gts = len(gt_boxes)
    if total_gts == 0:
        return 0.0, np.array([]), np.array([])

    if len(pred_boxes) == 0:
        return 0.0, np.zeros(1), np.zeros(1)

    # Sort predictions by confidence in descending order
    pred_boxes = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)

    # Keep track of True Positives (TP) and False Positives (FP)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))

    # To track matching, map image_id to list of ground truths for that image
    gt_by_image = {}
    for idx, gt in enumerate(gt_boxes):
        image_id = gt['image_id']
        if image_id not in gt_by_image:
            gt_by_image[image_id] = []
        # Add ground truth index to keep track of matched ones
        gt_by_image[image_id].append({'index': idx, 'box': gt['box'], 'matched': False})

    # Match predictions to ground truths
    for pred_idx, pred in enumerate(pred_boxes):
        image_id = pred['image_id']
        pred_box = pred['box']

        # If there are no ground truths in the predicted image, it's a false positive
        if image_id not in gt_by_image or len(gt_by_image[image_id]) == 0:
            fp[pred_idx] = 1
            continue

        # Find the ground truth box with the highest IoU in the same image
        best_iou = -1
        best_gt = None

        for gt in gt_by_image[image_id]:
            iou = calculate_iou(pred_box, gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        # If best IoU exceeds the threshold and the gt wasn't already matched
        if best_iou >= iou_threshold:
            if not best_gt['matched']:
                tp[pred_idx] = 1
                best_gt['matched'] = True
            else:
                # Already matched (duplicate detection)
                fp[pred_idx] = 1
        else:
            fp[pred_idx] = 1

    # Compute cumulative sums of TP and FP
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    # Compute Precision and Recall
    precisions = tp_cum / (tp_cum + fp_cum)
    recalls = tp_cum / total_gts

    # Standard Pascal VOC / COCO All-point Interpolation for AP
    # Add boundary values to precision and recall arrays
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    mrec = np.concatenate(([0.0], recalls, [1.0]))

    # Compute the envelope of precisions (monotonically decreasing)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find recall values where change occurs
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Integrate the area under Precision-Recall curve
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap, precisions, recalls


def calculate_map(all_gts, all_preds, iou_threshold=0.5):
    """
    Computes Mean Average Precision (mAP) across all classes.

    Parameters:
    - all_gts: dict mapping class_name -> list of gt_boxes
    - all_preds: dict mapping class_name -> list of pred_boxes
    - iou_threshold: float, IoU overlap threshold

    Returns:
    - map_score: float, mean average precision
    - class_aps: dict, class_name -> ap_value
    - pr_curves: dict, class_name -> (precisions, recalls)
    """
    class_aps = {}
    pr_curves = {}
    
    # We evaluate all classes present in either ground truths or predictions
    classes = sorted(list(set(all_gts.keys()) | set(all_preds.keys())))
    
    for cls in classes:
        gts = all_gts.get(cls, [])
        preds = all_preds.get(cls, [])
        
        ap, prec, rec = calculate_ap(gts, preds, iou_threshold)
        class_aps[cls] = ap
        pr_curves[cls] = (prec, rec)
        
    if len(class_aps) == 0:
        return 0.0, {}, {}
        
    map_score = sum(class_aps.values()) / len(class_aps)
    return map_score, class_aps, pr_curves


def plot_pr_curve(recalls, precisions, class_name, ap_value, save_path=None):
    """
    Plots the Precision-Recall curve and saves the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.step(recalls, precisions, color='b', alpha=0.8, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall Curve for {class_name} (AP={ap_value:.4f})')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_demo_evaluation():
    """
    Runs a test suite with synthetic data to verify calculations and demonstrate reports.
    """
    print("=" * 60)
    print("RUNNING EVALUATION FRAMEWORK DEMO (SYNTHETIC DATA)")
    print("=" * 60)
    
    # Define synthetic ground truths for classes: 'scratches', 'crazing', 'patches'
    synthetic_gts = {
        'scratches': [
            {'image_id': 1, 'box': [50, 50, 150, 150]},
            {'image_id': 1, 'box': [200, 200, 300, 300]},
            {'image_id': 2, 'box': [80, 80, 180, 180]}
        ],
        'crazing': [
            {'image_id': 2, 'box': [10, 10, 100, 100]},
            {'image_id': 3, 'box': [150, 150, 250, 250]}
        ],
        'patches': [
            {'image_id': 3, 'box': [100, 100, 200, 200]}
        ]
    }
    
    # Define synthetic predictions
    synthetic_preds = {
        'scratches': [
            # High confidence, exact match to gt 1 in image 1 (TP)
            {'image_id': 1, 'confidence': 0.95, 'box': [52, 48, 148, 152]},
            # Moderate confidence, match to gt 3 in image 2 (TP)
            {'image_id': 2, 'confidence': 0.88, 'box': [85, 85, 175, 175]},
            # Low confidence, duplicate detection of gt 1 in image 1 (FP)
            {'image_id': 1, 'confidence': 0.60, 'box': [50, 50, 145, 145]},
            # Very low confidence, completely wrong location (FP)
            {'image_id': 2, 'confidence': 0.30, 'box': [400, 400, 500, 500]}
        ],
        'crazing': [
            # High confidence, match to gt 1 in image 2 (TP)
            {'image_id': 2, 'confidence': 0.92, 'box': [12, 8, 98, 102]},
            # Low confidence, poor overlap with gt 2 in image 3 (FP - IoU < 0.5)
            {'image_id': 3, 'confidence': 0.45, 'box': [190, 190, 350, 350]}
        ],
        'patches': [] # No predictions for patches class (AP should be 0.0)
    }
    
    # Compute mAP at IoU = 0.5
    iou_thresh = 0.5
    map_score, class_aps, pr_curves = calculate_map(synthetic_gts, synthetic_preds, iou_threshold=iou_thresh)
    
    print("\nMetrics Calculation Results:")
    print("-" * 50)
    print(f"{'Class Name':<15} | {'Ground Truths':<13} | {'Predictions':<11} | {'AP (@0.5)':<10}")
    print("-" * 50)
    
    for cls in sorted(synthetic_gts.keys()):
        num_gts = len(synthetic_gts[cls])
        num_preds = len(synthetic_preds[cls])
        ap = class_aps.get(cls, 0.0)
        print(f"{cls:<15} | {num_gts:<13} | {num_preds:<11} | {ap:.4f}")
        
        # Save PR curves
        prec, rec = pr_curves[cls]
        if len(prec) > 0:
            save_path = os.path.join(REPORTS_DIR, f"{cls}_pr_curve.png")
            plot_pr_curve(rec, prec, cls, ap, save_path=save_path)
            print(f"  -> PR Curve saved to {save_path}")
            
    print("-" * 50)
    print(f"Mean Average Precision (mAP@{iou_thresh}): {map_score:.4f}")
    print("=" * 60)
    
    # Write a summary report
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Evaluation Metric Framework Verification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"mAP@{iou_thresh} Score: {map_score:.4f}\n\n")
        f.write("Class-wise AP:\n")
        for cls, ap in class_aps.items():
            f.write(f"  - {cls}: {ap:.4f} (GTs: {len(synthetic_gts[cls])}, Preds: {len(synthetic_preds[cls])})\n")
    print(f"Summary report written to {report_path}")


if __name__ == "__main__":
    run_demo_evaluation()
