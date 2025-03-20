import numpy as np

"""
mAP contribution of minority classes is too low.
Instead of mAP, we compute AP for each class separately over each image.
We keep track of the number of images that contribute to the average precision for each class.
We compute the precision and recall for each class and then compute the average precision using 11-point interpolation.
We sum up the individual average precision of each class over all images.
We divide the sum by the number of images that contribute to the average precision for each class.
"""


def compute_iou(ground_truth_boxes, predicted_boxes):
    """
    args:
        ground_truth_boxes: (N, 4) array containing the coordinates
                            of the ground truth boxes
        predicted_boxes:    (M, 4) array containing the coordinates
                            of the predicted boxes
    returns:
        iou:                (N, M) array containing the intersection over
                            union of the ground truth boxes and the predicted boxes.
    """
    N, M = len(ground_truth_boxes), len(predicted_boxes)
    xMin2, yMin2, xMax2, yMax2 = predicted_boxes.T
    intersectingRegions = np.zeros((N, M, 4), dtype=int)
    intersectingAreas = np.zeros((N, M))
    unionAreas = np.zeros((N, M))

    for i in range(N):
        xMin1, yMin1, xMax1, yMax1 = ground_truth_boxes[i]
        xMin = np.maximum(xMin1, xMin2)[:, np.newaxis]
        xMax = np.minimum(xMax1, xMax2)[:, np.newaxis]
        yMin = np.maximum(yMin1, yMin2)[:, np.newaxis]
        yMax = np.minimum(yMax1, yMax2)[:, np.newaxis]
        intersectingRegions[i] = np.concatenate((xMin, yMin, xMax, yMax), axis=1)
        intersectingAreas[i] = (np.maximum(xMax - xMin, 0) * np.maximum(yMax - yMin, 0)).ravel()
        unionAreas[i] = np.maximum(xMax2 - xMin2, 0) * np.maximum(yMax2 - yMin2, 0) + \
                        np.maximum(xMax1 - xMin1, 0) * np.maximum(yMax1 - yMin1, 0) - \
                            intersectingAreas[i]
        unionAreas[i] = np.maximum(unionAreas[i], 0)
    iou = np.nan_to_num(intersectingAreas / unionAreas)

    return iou


def precision_recall_curve(ground_truth_boxes,
                            ground_truth_labels,
                            positive_class,
                            predicted_boxes,
                            predicted_labels,
                            scores,
                            iou_threshold):
    """
    args:
        ground_truth_boxes:     Array containing the coordinates
                                of the ground truth boxes
        ground_truth_labels:    Array containing the labels of the ground truth boxes
        positive_class:         The label of the positive class
        predicted_boxes:        Array containing the coordinates
                                of the predicted boxes
        predicted_labels:       Array containing the labels of the predicted boxes
        scores:                 Array containing the confidence scores of the predicted boxes
        iou_threshold:          The threshold for considering a prediction to be a true positive
    returns:
        precisions:             The precision values for the given recall values
        recalls:                The recall values for the given precision values
    """
    # Filter out the boxes and labels for the positive class
    class_gt_boxes = ground_truth_boxes[ground_truth_labels == positive_class]
    class_predicted_boxes = predicted_boxes[predicted_labels == positive_class]
    scores = scores[predicted_labels == positive_class]
    contributes = 1 # To check if positive class in current image contributes to AP
    if len(class_gt_boxes) == 0: # No ground truth boxes for this class
        if len(class_predicted_boxes) == 0: # No predicted boxes either
            contributes = 0 # No contribution to AP
        return np.array([0]), np.array([0]), contributes
    
    iou = compute_iou(class_gt_boxes, class_predicted_boxes)
    N, M = iou.shape
    # scores need global sorting because they are sorted 
    # according to class labels (locally sorted)
    sorted_score_indices = np.argsort(scores)[::-1]
    matched_boxes = set()
    gt_counter = 0
    predicted_box_counter = 0
    TP, FP = 0, 0
    TPs, FPs = [], []
    
    while len(matched_boxes) < N and predicted_box_counter < M:
        predicted_box_index = sorted_score_indices[predicted_box_counter]
        highest_iou_gt_box_indices = \
            np.argsort(iou[:, predicted_box_index])[::-1]
        # If the highest IOU is less than threshold, skip predicted box
        if iou[highest_iou_gt_box_indices[gt_counter], \
               predicted_box_index] < iou_threshold:
            predicted_box_counter += 1
            gt_counter = 0
            FP += 1
            TPs.append(TP)
            FPs.append(FP)
            continue
        gt_box_matched = highest_iou_gt_box_indices[gt_counter]
        if gt_box_matched not in matched_boxes:
            matched_boxes.add(gt_box_matched)
            predicted_box_counter += 1
            gt_counter = 0
            TP += 1
            TPs.append(TP)
            FPs.append(FP)
        else:
            gt_counter += 1

    FP = M - TP
    TPs.append(TP)
    FPs.append(FP)
    TPs, FPs = np.array(TPs), np.array(FPs)
    precisions = np.nan_to_num(TPs / (TPs + FPs))
    recalls = TPs / N
    
    return precisions, recalls, contributes


def ap_11pt(precisions, recalls):
    """
    args:
        precisions: The precision values sorted according to the recall values.
        recalls:    The sorted recall values.
    returns:
        ap:         The average precision computed using 11-point interpolation.
    """
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if (recalls >= t).sum() == 0:
            p = 0
        else:
            p = precisions[recalls >= t].max()
        ap += p
    ap /= 11

    return ap