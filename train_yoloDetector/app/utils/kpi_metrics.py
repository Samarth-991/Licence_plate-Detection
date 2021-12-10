import numpy as np


def calc_iou(gt_bbox, pred_bbox):
    '''
    This function takes the predicted bounding box and ground truth bounding box and
    return the IoU ratio
    '''
    xmin_gt, ymin_gt,xmax_gt, ymax_gt = gt_bbox
    xmin_p, ymin_p, xmax_p, ymax_p = pred_bbox
    if (xmin_gt > ( xmax_gt)) or (ymin_gt > (ymax_gt)):
        raise AssertionError("Ground Truth Bounding Box is not correct")
    if (xmin_p > (xmax_p)) or (ymin_p > (ymax_p)):
        raise AssertionError("Predicted Bounding Box is not correct")

    # if the GT bbox and predcited BBox do not overlap then iou=0
    if (xmax_p < xmin_p):
        # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        return 0.0
    if (ymax_gt  < ymin_p):
        # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        return 0.0
    if (xmin_gt > xmax_p):
        # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        return 0.0
    if (ymin_gt > ymax_p):
        # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        return 0.0
    GT_bbox_area = (xmax_gt-xmin_gt + 1) * (ymax_gt-ymin_gt + 1)
    Pred_bbox_area = (xmax_p-xmin_p + 1) * (ymax_p-ymin_gt + 1)

    x_top_left = np.max([xmin_gt, xmin_p])
    y_top_left = np.max([ymin_gt, ymin_p])
    x_bottom_right = np.min([xmax_gt, xmax_p])
    y_bottom_right = np.min([ymax_gt, ymax_p])
    intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

    union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
    return float(intersection_area / union_area)


def get_metrices(imgname,gt_data,pred_data,iou_thr=0.5):
    '''
        This function takes the image corresponding prediction bounding boxes
        and ground truth bounding boxes to generate KPI Metrics for precision and recall
    '''
    gt_idx_thr = []
    pred_idx_thr = []
    ious=[]
    gt_cords = gt_data[imgname]
    pred_cords = pred_data[imgname]
    if len(range(len(gt_cords))) == 0:
        return {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    if len(range(len(gt_cords))) == 0:
        return {'true_positive': 0, 'false_positive': 0, 'false_negative':0}

    for ipb, pred_box in enumerate(pred_cords[0]):

        for igb, gt_box in enumerate(gt_cords[0]):
            iou = calc_iou(gt_box, pred_box[:4])
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)
    iou_sort = np.argsort(ious)[::1]
    if len(iou_sort)==0:
        return {'true_positive': 0, 'false_positive': 0, 'false_negative': 0}
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in iou_sort:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_cords[0]) - len(pred_match_idx)
        fn = len(gt_cords[0]) - len(gt_match_idx)

    return {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}


def calc_precision_recall(image_results):
    """Calculates precision and recall from the set of images
    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
            }
    Returns:
        tuple: of floats of (precision, recall)
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    precisions = []
    recalls = []
    for img_id, res in image_results.items():
        true_positive += res[0]['true_positive']
        false_positive += res[0]['false_positive']
        false_negative += res[0]['false_negative']

        try:
            precisions.append(true_positive / (true_positive + false_positive))
        except ZeroDivisionError:
            precision = 0.0
        try:
            recalls.append(true_positive / (true_positive + false_negative))
        except ZeroDivisionError:
            recall = 0.0

    precisions = np.array(list(map(lambda x: round(x * 100, 2), precisions)))
    recalls = np.array(list(map(lambda x: round(x* 100, 2), recalls)))

    return (precisions, recalls)
