def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU value
    """
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    box1_area = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    box2_area = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)

    intersection = max(0, min(xmax1, xmax2) - max(xmin1, xmin2) + 1) * max(0, min(ymax1, ymax2) - max(ymin1, ymin2) + 1)

    iou = intersection / (box1_area - intersection + box2_area)
    return iou