import os
import torch
from typing import Tuple
from src.training.retinanet_trainer import retinanet_training
from src.training.metrics import *
from src.plots.bounding_box import image_with_bounding_box
import matplotlib.pyplot as plt
import gc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def save_model(model, optimizer, saved_model_dir) -> None:
    """
    Saves the model state and optimizer state on the dict
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(saved_model_dir, "checkpoint.pt"),
    )


def train(train_loader, model, optimizer, dx, device) -> Tuple[float, float]:
    num_classes = model.num_classes
    """Implements the main training loop."""
    model.train()

    train_loss_meter = AverageMeter("train loss")
    train_classification_loss_meter = AverageMeter("train loss")
    train_regression_loss_meter = AverageMeter("train loss")
    train_map_meter = AverageMeter("train accuracy")
    thresholds = np.arange(0.1, 0.3, dx)

    # loop over each minibatch
    for batchCount, (images, boxes_labels) in enumerate(train_loader):
        # print(f"Starting batch {batchCount}\n")
        images = images.to(device)
        boxes_labels = boxes_labels.to(device)

        n = images.shape[0]
        features, classification, regression, anchors = model(images)

        batch_loss, results = retinanet_training(model, images, classification, regression, anchors, boxes_labels, "training", device)
        classification_loss, regression_loss = batch_loss
        classification_loss, regression_loss = classification_loss[0], regression_loss[0]
        batch_loss = batch_loss[0] + batch_loss[1]
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()

        # Return data to cpu
        images = images.cpu()
        boxes_labels = boxes_labels.cpu().numpy()
        boxes = boxes_labels[:, :, :4]
        true_labels = boxes_labels[:, :, 4]

        for f in range(len(features)):
            features[f] = features[f].detach().cpu()
        classification = classification.cpu()
        regression = regression.cpu()
        anchors = anchors.cpu()
        batch_loss = batch_loss.cpu()

        mean_ap = 0
        for i in range(n):
            for key in results[i]:
                results[i][key] = results[i][key].detach().cpu()
            finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates = results[i]["scores"], results[i]["labels"], results[i]["boxes"]
            finalAnchorBoxesCoordinates = torch.maximum(finalAnchorBoxesCoordinates, torch.tensor(0)).int()
            if finalAnchorBoxesCoordinates.ndim > 1: # If a prediction is made
                # iou = compute_iou(boxes[i], finalAnchorBoxesCoordinates.numpy())
                # matched_boxes = match_boxes(iou, 0.1)
                # valid_box_indices = []
                # matched_true_boxes = []
                # for key in matched_boxes:
                #     if matched_boxes[key] != -1:
                #         matched_true_boxes.append(key)
                #         valid_box_indices.append(matched_boxes[key])
                # print(true_labels[i][matched_true_boxes])
                # bbox_image = image_with_bounding_box(images[i], finalAnchorBoxesCoordinates[valid_box_indices])
                # plt.imshow(bbox_image)
                # plt.show()
                # break
                iou = compute_iou(boxes[i], finalAnchorBoxesCoordinates)
                # matched_boxes = match_boxes(iou)
                precisions = np.zeros((num_classes, len(thresholds)))
                recalls = np.zeros((num_classes, len(thresholds)))
                for k in range(num_classes):
                    precisions[k], recalls[k] = precision_recall_curve(true_labels[i], finalAnchorBoxesIndexes, iou, k, thresholds)
                mean_ap += compute_mean_average_precision(precisions, recalls, dx)

        mean_ap /= n

        # Save loss

        train_loss_meter.update(val=float(batch_loss.item()), n=n)
        train_classification_loss_meter.update(val=float(classification_loss.item()), n=n)
        train_regression_loss_meter.update(val=float(regression_loss.item()), n=n)
        train_map_meter.update(val=mean_ap, n=n)

        del images, boxes_labels, classification_loss, regression_loss, batch_loss, features, classification, regression, anchors, results
        gc.collect()
        torch.cuda.empty_cache()
        # break

    return train_classification_loss_meter.avg, train_regression_loss_meter.avg, train_loss_meter.avg, train_map_meter.avg


def validate(val_loader, model, optimizer, device) -> Tuple[float, float]:
    """Evaluate on held-out split (either val or test)"""
    model.eval()

    val_loss_meter = AverageMeter("train loss")
    train_map_meter = AverageMeter("train accuracy")
    thresholds = np.arange(0.3, 0.7, dx)

    # loop over each minibatch
    for batchCount, (images, boxes_labels) in enumerate(val_loader):
        # print(f"Starting batch {batchCount}\n")
        images = images.to(device)
        boxes_labels = boxes_labels.to(device)

        n = images.shape[0]
        features, classification, regression, anchors = model(images)

        batch_loss, finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates = retinanet_training(model, images, classification, regression, anchors, boxes_labels, "training")
        batch_loss = batch_loss[0] + batch_loss[1]
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Return data to cpu
        images = images.cpu()
        boxes_labels = boxes_labels.cpu()
        boxes = boxes_labels[:, :, :4]
        labels = boxes_labels[:, :, 4]

        # features = features.cpu()
        classification = classification.cpu()
        regression = regression.cpu()
        anchors = anchors.cpu()
        batch_loss = batch_loss.cpu()
        finalScores = finalScores.detach().cpu().numpy()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.detach().cpu()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.detach().cpu().numpy()

        # Save loss

        val_loss_meter.update(val=float(batch_loss.item()), n=n)
    return val_loss_meter.avg, val_acc_meter.avg