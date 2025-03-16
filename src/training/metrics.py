from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


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


def match_boxes(iou, scores,
                confidence_threshold,
                iou_threshold):
    """
    args:
        iou:                    The intesection over union of the ground truth boxes
                                and predicted boxes. (N x M).
        confidence_threshold:   The minimum confidence required for a predicted box to match
                                with a ground truth box.                
        iou_threshold:          The minimum iou required for a predicted box to match
                                with a ground truth box.
    returns:
        matched_boxes:          dictionary containing indices of ground truth boxes
                                mapped to the indices of predicted boxes.
    """
    N, M = iou.shape
    matched_boxes = {} # {gt_index: predicted_index}
    gt_counter = 0
    predicted_box_index = 0
    visited_predicted_boxes = set()
    while len(matched_boxes) < N and predicted_box_index < M and \
        scores[predicted_box_index] >= confidence_threshold:
        if predicted_box_index not in visited_predicted_boxes:
            highest_iou_gt_box_indices = np.argsort(iou[:, predicted_box_index])[::-1]
            visited_predicted_boxes.add(predicted_box_index)
        # If the highest IOU is less than threshold, skip predicted box
        if iou[highest_iou_gt_box_indices[gt_counter], \
               predicted_box_index] < iou_threshold:
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


def extract_predicted_labels(matched_boxes, all_predicted_labels):
    """
    args:
        matched_boxes:          dictionary containing indices of ground truth boxes
                                mapped to the indices of predicted boxes.
        all_predicted_labels:   The labels of all the predicted boxes.
    returns:
        predicted_labels:       Predicted labels of the predicted boxes that matched
                                with ground truth boxes.
    """
    predicted_labels = np.zeros(len(matched_boxes), dtype=int) - 1
    for gt_box_index in matched_boxes:
        if matched_boxes[gt_box_index] != -1:
            predicted_labels[gt_box_index] = all_predicted_labels[matched_boxes[gt_box_index]]
    
    return predicted_labels


def precision_recall_curve(true_labels,
                           all_predicted_labels,
                           scores, iou, positive_class,
                           confidence_thresholds,
                           iou_threshold,
                           placeholder_class=100):
    """
    For a given class labels, finds the precision-recall curve over different thresholds
    of the output score.
    args:
        true_labels:            Ground-truth labels of the bounding boxes.
        all_predicted_labels:   The labels of all the predicted boxes.
        positive class:         The class in consideration (1-vs-all).
        confidence_thresholds:  The minimum confidence required for a predicted box
                                to match with a ground truth box.
        iou_threshold:          The minimum iou required for a predicted box
                                to match with a ground truth box.
        placeholder_class:      A placeholder for a simple swap.
    returns:
        precisions:             The precision values over all thresholds.
        recalls:                The recall values over all thresholds.
    """
    precisions, recalls = [], []
    true_labels_copy = true_labels.copy()
    true_labels_copy[true_labels_copy == positive_class] = placeholder_class
    true_labels_copy[true_labels_copy != placeholder_class] = 0
    true_labels_copy[true_labels_copy == placeholder_class] = 1
    for confidence_threshold in confidence_thresholds:
        matched_boxes = match_boxes(iou, scores, confidence_threshold, iou_threshold)
        predicted_labels = extract_predicted_labels(matched_boxes, all_predicted_labels)
        predicted_labels[predicted_labels == positive_class] = placeholder_class
        predicted_labels[predicted_labels != placeholder_class] = 0
        predicted_labels[predicted_labels == placeholder_class] = 1
        # print(true_labels_copy, predicted_labels)
        precisions.append(precision_score(y_true=true_labels_copy, y_pred=predicted_labels, pos_label=1))
        recalls.append(recall_score(y_true=true_labels_copy, y_pred=predicted_labels, pos_label=1))
    precisions, recalls = np.array(precisions), np.array(recalls)
    sort_indices = np.argsort(recalls)

    return precisions[sort_indices], recalls[sort_indices]


def trapezoid_rule(precisions, recalls):
    areas = (recalls[1:] - recalls[:-1]) * (precisions[1:] + precisions[:-1]) / 2

    return areas.sum()


def compute_mean_average_precision(precisions, recalls):
    """
    Computes the area under the precision-recall curve.
    args:
        precisions: (K, steps) array containing class-wise precision scores,
                    where K is the number of classes and steps is the
                    number of thresholds.
        recalls:    (K, steps) array containing class-wise recall scores,
                    where K is the number of classes and steps is the
                    number of thresholds.
    """
    K = len(precisions)
    average_precisions = np.zeros(K)
    for k in range(K):
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
    precision = precision_score(y_true=true_labels, y_pred=predicted_labels, pos_label=positive_class)
    recall = recall_score(y_true=true_labels, y_pred=predicted_labels, pos_label=positive_class)
    f1_score = (2 * precision * recall / (precision + recall)).nan_to_num(0)

    return precision, recall, f1_score