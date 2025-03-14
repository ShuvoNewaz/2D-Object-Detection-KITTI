import torch
# from torchvision.ops import nms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/ponta256/fssd-resnext-voc-coco/blob/master/layers/box_utils.py#L245
def nms(boxes, scores, nms_thresh=0.5, top_k=200):
    boxes_copy = boxes.clone().detach().cpu().numpy()
    scores_copy = scores.clone().detach().cpu().numpy()
    keep = []
    if len(boxes) == 0:
        return keep
    x1 = boxes_copy[:, 0]
    y1 = boxes_copy[:, 1]
    x2 = boxes_copy[:, 2]
    y2 = boxes_copy[:, 3]
    area = (x2-x1)*(y2-y1)
    idx = np.argsort(scores_copy, axis=0)   # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals


    while len(idx) > 0:
        last = len(idx)-1
        i = idx[last]  # index of current largest val
        keep.append(i)
  
        xx1 = np.maximum(x1[i], x1[idx[:last]])
        yy1 = np.maximum(y1[i], y1[idx[:last]])
        xx2 = np.minimum(x2[i], x2[idx[:last]])
        yy2 = np.minimum(y2[i], y2[idx[:last]])

        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)

        inter = w*h
        iou = inter / (area[idx[:last]]+area[i]-inter)
        idx = np.delete(idx, np.concatenate(([last], np.where(iou > nms_thresh)[0])))

    return np.array(keep, dtype=np.int64)

# https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py#L39
def batched_nms(
    boxes,
    scores,
    idxs,
    iou_threshold,
):

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = nms(boxes_for_nms, scores, nms_thresh=iou_threshold)
        return keep


def retinanet_outputs(model, img_batch, classification, regression, anchors):
    transformed_anchors = model.regressBoxes(anchors, regression)
    transformed_anchors = model.clipBoxes(transformed_anchors, img_batch)
    predicted_labels = torch.argmax(classification, dim=-1)

    results = []

    for i in range(len(classification)):
        results.append({})
        finalScores = torch.Tensor([]).to(device)
        finalPredictedLabels = torch.Tensor([]).to(device)
        finalAnchorBoxesCoordinates = torch.Tensor([]).to(device)
        for k in range(classification.shape[2]):
            scores = torch.squeeze(classification[i, :, k])
            scores_over_thresh = (scores > 0.05)
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just continue
                continue
            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(transformed_anchors[i])
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
           
            finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
            finalPredictedLabels = torch.cat((finalPredictedLabels, predicted_labels[i, anchors_nms_idx]))
            finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        results[i]["scores"] = finalScores
        results[i]["labels"] = finalPredictedLabels
        results[i]["boxes"] = finalAnchorBoxesCoordinates

    return results