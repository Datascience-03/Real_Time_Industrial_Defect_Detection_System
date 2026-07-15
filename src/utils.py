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
