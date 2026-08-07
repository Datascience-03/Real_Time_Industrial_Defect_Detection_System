def yolo_to_voc(x_center, y_center, w, h, img_width, img_height):
    """
    Converts YOLO format bounding box (normalized center x, center y, width, height)
    to Pascal VOC format (pixel coordinates xmin, ymin, xmax, ymax).
    """
    xmin = int((x_center - w / 2) * img_width)
    ymin = int((y_center - h / 2) * img_height)
    xmax = int((x_center + w / 2) * img_width)
    ymax = int((y_center + h / 2) * img_height)

    # Clip coordinates to image boundary
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))

    return [xmin, ymin, xmax, ymax]


def voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    Converts Pascal VOC format (pixel coordinates xmin, ymin, xmax, ymax)
    to YOLO format (normalized center x, center y, width, height).
    """
    # Ensure coordinates are within image boundaries
    xmin = max(0, min(xmin, img_width))
    ymin = max(0, min(ymin, img_height))
    xmax = max(0, min(xmax, img_width))
    ymax = max(0, min(ymax, img_height))

    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2
    y_center = ymin + h / 2

    # Normalize by image dimensions
    return [x_center / img_width, y_center / img_height, w / img_width, h / img_height]


def compute_iou(box1, box2):
    """
    Computes Intersection over Union (IoU) between two bounding boxes
    in Pascal VOC format [xmin, ymin, xmax, ymax].

    Parameters:
        box1 (list): [xmin, ymin, xmax, ymax] for the first box.
        box2 (list): [xmin, ymin, xmax, ymax] for the second box.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    # Intersection rectangle
    inter_xmin = max(box1[0], box2[0])
    inter_ymin = max(box1[1], box2[1])
    inter_xmax = min(box1[2], box2[2])
    inter_ymax = min(box1[3], box2[3])

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    intersection = inter_w * inter_h

    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def filter_detections_by_confidence(detections, threshold=0.5):
    """
    Filters a list of detection dictionaries by a minimum confidence threshold.

    Parameters:
        detections (list): List of dicts, each containing a 'confidence' key.
        threshold  (float): Minimum confidence score to keep (default: 0.5).

    Returns:
        list: Filtered list of detections above the threshold.
    """
    return [d for d in detections if d.get("confidence", 0) >= threshold]


def format_detection_label(class_name, confidence):
    """
    Formats a detection label string for annotation overlays.

    Parameters:
        class_name (str): The predicted defect class name.
        confidence (float): Confidence score (0.0 to 1.0).

    Returns:
        str: Formatted label, e.g. "crazing 94.5%".
    """
    return f"{class_name} {confidence * 100:.1f}%"


def compute_bbox_area(box):
    """
    Computes the area of a bounding box in Pascal VOC format.

    Parameters:
        box (list): [xmin, ymin, xmax, ymax]

    Returns:
        int: Area of the bounding box in pixels.
    """
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h


def is_valid_bbox(box, img_width, img_height):
    """
    Validates that a bounding box is within image bounds and has positive area.

    Parameters:
        box        (list): [xmin, ymin, xmax, ymax]
        img_width  (int):  Image width in pixels.
        img_height (int):  Image height in pixels.

    Returns:
        bool: True if the box is valid, False otherwise.
    """
    xmin, ymin, xmax, ymax = box
    if xmin < 0 or ymin < 0:
        return False
    if xmax > img_width or ymax > img_height:
        return False
    if xmax <= xmin or ymax <= ymin:
        return False
    return True
