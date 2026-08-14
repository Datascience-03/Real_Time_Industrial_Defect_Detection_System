import os
import argparse
import cv2
from ultralytics import YOLO
from src.evaluate import calculate_map, plot_pr_curve
from src.utils import get_checkpoint_path


def load_ground_truths(images_dir, labels_dir):
    all_gts = {}
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        h, w = cv2.imread(img_path).shape[:2]
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        if not os.path.exists(label_path):
            continue
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                xc, yc, ww, hh = map(float, parts[1:5])
                # convert normalized xywh to xyxy
                x_c = xc * w
                y_c = yc * h
                box_w = ww * w
                box_h = hh * h
                xmin = max(0, x_c - box_w / 2)
                ymin = max(0, y_c - box_h / 2)
                xmax = min(w, x_c + box_w / 2)
                ymax = min(h, y_c + box_h / 2)
                if cls not in all_gts:
                    all_gts[cls] = []
                all_gts[cls].append({'image_id': img_name, 'box': [xmin, ymin, xmax, ymax]})
    return all_gts


def extract_predictions_from_result(result, img_name):
    preds = []
    # Attempt multiple attribute access patterns for different runtimes
    try:
        boxes = result.boxes
        # boxes may be iterable
        for box in boxes:
            try:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                # xyxy
                try:
                    xy = box.xyxy[0]
                    xy = [float(xy[0]), float(xy[1]), float(xy[2]), float(xy[3])]
                except Exception:
                    # fallback to .xyxy if different shape
                    xy = [float(x) for x in box.xyxy]
                preds.append({'image_id': img_name, 'confidence': confidence, 'box': xy, 'class': str(cls_id)})
            except Exception:
                continue
    except Exception:
        pass
    return preds


def run_evaluation_for_model(model_path, images_dir, labels_dir, iou=0.5):
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))])

    all_preds = {}
    # Collect predictions per class
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        results = model.predict(source=img_path, conf=0.001, iou=0.45, imgsz=640, verbose=False)
        for res in results:
            preds = extract_predictions_from_result(res, img_name)
            for p in preds:
                cls = p['class']
                if cls not in all_preds:
                    all_preds[cls] = []
                all_preds[cls].append({'image_id': p['image_id'], 'confidence': p['confidence'], 'box': p['box']})

    all_gts = load_ground_truths(images_dir, labels_dir)

    map_score, class_aps, pr_curves = calculate_map(all_gts, all_preds, iou_threshold=iou)

    print(f"mAP@{i/100 if False else iou}: {map_score:.4f}")
    report_dir = 'outputs/reports'
    os.makedirs(report_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    report_path = os.path.join(report_dir, f"evaluation_{model_name}.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"mAP@{iou}: {map_score:.4f}\n")
        for cls, ap in class_aps.items():
            f.write(f"Class {cls}: AP={ap:.4f}\n")
    print(f"Wrote evaluation report to {report_path}")

    # Save PR curves for each class
    for cls, (prec, rec) in pr_curves.items():
        if len(prec) > 0:
            save_path = os.path.join(report_dir, f"{model_name}_class_{cls}_pr.png")
            plot_pr_curve(rec, prec, f"class_{cls}", class_aps.get(cls, 0.0), save_path=save_path)

    return map_score, class_aps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', '-m', nargs='+', default=['runs/detect/train/weights/best.pt','runs/detect/train/weights/best.onnx'], help='Model paths to evaluate')
    parser.add_argument('--images', default='dataset/test/images', help='Path to test images')
    parser.add_argument('--labels', default='dataset/test/labels', help='Path to test labels')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for mAP')
    args = parser.parse_args()

    results = {}
    for m in args.models:
        chosen = m
        if m.endswith('best.pt'):
            ck = get_checkpoint_path()
            if ck is not None:
                chosen = ck
            else:
                print(f"No checkpoint found in runs/.../weights, using {m} path as-is")
        if not os.path.exists(chosen):
            print(f"Model not found: {chosen}, skipping")
            continue
        map_score, class_aps = run_evaluation_for_model(chosen, args.images, args.labels, iou=args.iou)
        results[chosen] = {'map': map_score, 'class_aps': class_aps}

    # Summary
    summary_path = 'outputs/reports/evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        for m, r in results.items():
            f.write(f"Model: {m}\n")
            f.write(f"  mAP@{args.iou}: {r['map']:.4f}\n")
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
