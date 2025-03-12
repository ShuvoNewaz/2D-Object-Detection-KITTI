from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def compute_iou(ground_truth_boxes, predicted_boxes):
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


def match_boxes(iou, threshold):
    # iou[np.where(iou < threshold)] = 0
    N, M = iou.shape
    matched_boxes = {} # {gt_index: predicted_index}
    gt_counter = 0
    predicted_box_index = 0
    while len(matched_boxes) < N and predicted_box_index < M:
        highest_iou_gt_box_indices = np.argsort(iou[:, predicted_box_index])[::-1]

        # If the highest IOU is less than threshold, skip predicted box
        if iou[highest_iou_gt_box_indices[gt_counter], predicted_box_index] < threshold:
            predicted_box_index += 1
            gt_counter = 0
            continue
        gt_box_matched = highest_iou_gt_box_indices[gt_counter]
        if gt_box_matched not in matched_boxes:
            matched_boxes[gt_box_matched] = predicted_box_index
            predicted_box_index += 1
            gt_counter = 0
        else:
            gt_counter += 1

    # Fill up un-predicted ground truth boxes with garbage predicted box indices
    for i in range(N):
        if i not in matched_boxes:
            matched_boxes[i] = -1
    return matched_boxes


def extract_predicted_labels(matched_boxes, true_labels, all_predicted_labels):
    predicted_labels = np.zeros(true_labels.shape, dtype=int) - 1
    for gt_box_index in matched_boxes:
        if matched_boxes[gt_box_index] != -1:
            predicted_labels[gt_box_index] = all_predicted_labels[matched_boxes[gt_box_index]]
    
    return predicted_labels


def precision_recall_curve(true_labels, all_predicted_labels, iou, positive_class, thresholds, placeholder_class=100):
    """
    For a given class labels, finds the precision-recall curve over different thresholds
    of the output score.
    args:
        true_labels:        Ground-truth labels of the bounding boxes.
        predicted_scores:   Scores of the predicted bounding boxes.
        positive class:     The class in consideration (1-vs-all).
        thresholds:         The threshold values which determine positive/negative classes
    """
    precisions, recalls = [], []
    true_labels_copy = true_labels.copy()
    true_labels_copy[true_labels_copy == positive_class] = placeholder_class
    true_labels_copy[true_labels_copy != placeholder_class] = 0
    true_labels_copy[true_labels_copy == placeholder_class] = 1
    for threshold in thresholds:
        matched_boxes = match_boxes(iou, threshold)
        predicted_labels = extract_predicted_labels(matched_boxes, true_labels, all_predicted_labels)
        predicted_labels[predicted_labels == positive_class] = placeholder_class
        predicted_labels[predicted_labels != placeholder_class] = 0
        predicted_labels[predicted_labels == placeholder_class] = 1
        precisions.append(precision_score(y_true=true_labels_copy, y_pred=predicted_labels, pos_label=1))
        recalls.append(recall_score(y_true=true_labels_copy, y_pred=predicted_labels, pos_label=1))
    precisions, recalls = np.array(precisions), np.array(recalls)
    sort_indices = np.argsort(recalls)

    return precisions[sort_indices], recalls[sort_indices]


def trapezoid_rule(precisions, recalls):
    areas = (recalls[1:] - recalls[:-1]) * precisions[1:]

    return areas.sum()


def compute_mean_average_precision(precisions, recalls, dx):
    """
    Computes the area under the precision-recall curve.
    args:
        precisions: (K, steps) array containing class-wise precision scores,
                    where K is the number of classes and steps is the
                    number of thresholds.
        recalls:    (K, steps) array containing class-wise recall scores,
                    where K is the number of classes and steps is the
                    number of thresholds.
        dx:         Threshold increment size.
    """
    K = len(precisions)
    average_precisions = np.zeros(K)
    for k in range(K):
        # average_precisions[k] = np.trapz(y=precisions[k], x=recalls[k], dx=dx)
        average_precisions[k] = trapezoid_rule(precisions[k], recalls[k])

    return average_precisions.mean()


def compute_f1_score(true_labels, predicted_labels, positive_class: int):
    """
        args:
            true_labels: torch tensor containing the ground truth labels of objects.
            predicted_labels: torch tensor containing the predicted labels of objects.
            positive_class: The class for which the scores are being computed (1-vs-all).
        returns:
            precision
            recall
            f1_score
    """
    # gt_label = true_labels.copy()
    # pred_label = predicted_labels.copy()
    # gt_label[gt_label != positive_class] = 0
    # pred_label[pred_label != positive_class] = 0

    # recall = torch.mean((pred_label[gt_label == 1] == 1).float())
    # precision = torch.mean((1 == gt_label[pred_label == 1]).float())
    precision = precision_score(y_true=true_labels, y_pred=predicted_labels, pos_label=positive_class)
    recall = recall_score(y_true=true_labels, y_pred=predicted_labels, pos_label=positive_class)
    f1_score = (2*precision*recall / (precision + recall)).nan_to_num(0)

    return precision, recall, f1_score