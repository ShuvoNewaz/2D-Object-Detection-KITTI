import os
import torch
from typing import Tuple
from src.models.retinanet.outputs import retinanet_outputs
from src.training.metrics import *
from src.plots.bounding_box import image_with_bounding_box
import matplotlib.pyplot as plt
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from src.plots.bounding_box import image_with_bounding_box
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


def epoch_runner(loader, model, optimizer, dx, device):
    """Implements the main training/validation loop."""
    num_classes = model.num_classes
    loss_meter = AverageMeter("train loss")
    classification_loss_meter = AverageMeter("train loss")
    regression_loss_meter = AverageMeter("train loss")
    map_meter = AverageMeter("train accuracy")
    confidence_thresholds = np.arange(0.05, 0.95, dx)

    # loop over each minibatch
    for batchCount, (images, boxes_labels) in enumerate(loader):
        images = images.to(device)
        boxes_labels = boxes_labels.to(device)

        n = images.shape[0]
        features, classification, regression, anchors = model(images)
        batch_loss = model.loss_criterion(classification, regression, anchors, boxes_labels)
        with torch.no_grad():
            results = retinanet_outputs(model, images, classification, regression, anchors)
        classification_loss, regression_loss = batch_loss
        classification_loss, regression_loss = classification_loss[0], regression_loss[0]
        batch_loss = batch_loss[0] + 1 * batch_loss[1]

        if optimizer:
            optimizer.zero_grad(set_to_none=True)
            batch_loss.backward()
            optimizer.step()

        # Free up GPU
        batch_loss = batch_loss.detach().cpu().item()
        classification_loss = classification_loss.detach().cpu().item()
        regression_loss = regression_loss.detach().cpu().item()
        images = images.cpu()
        boxes_labels = boxes_labels.cpu().numpy()
        boxes = boxes_labels[:, :, :4]
        true_labels = boxes_labels[:, :, 4]
        
        for f in range(len(features)):
            features[f] = features[f].detach().cpu()
        classification = classification.detach().cpu()
        regression = regression.detach().cpu()
        anchors = anchors.detach().cpu()
        del features, classification, regression, anchors

        # Compute MAP
        with torch.no_grad():
            mean_ap = 0
            result_keys = ["scores", "labels", "boxes"]
            for i in range(n):
                for key in result_keys:
                    results[i][key] = results[i][key].detach().cpu()
                finalScores, finalPredictedLabels, finalAnchorBoxesCoordinates = \
                    results[i]["scores"], results[i]["labels"], results[i]["boxes"]
                results[i].clear()
                finalAnchorBoxesCoordinates = torch.maximum(finalAnchorBoxesCoordinates, torch.tensor(0)).int()
                if finalAnchorBoxesCoordinates.ndim > 1: # If a prediction is made
                    iou = compute_iou(boxes[i], finalAnchorBoxesCoordinates)
                    precisions = np.zeros((num_classes, len(confidence_thresholds)))
                    recalls = np.zeros((num_classes, len(confidence_thresholds)))
                    for k in range(num_classes):
                        precisions[k], recalls[k] = \
                            precision_recall_curve(true_labels[i].astype(int),
                                                finalPredictedLabels.numpy().astype(int),
                                                finalScores,
                                                iou, k,
                                                confidence_thresholds,
                                                iou_threshold=0.5)
                    mean_ap += compute_mean_average_precision(precisions, recalls)
                
        mean_ap /= n

        # Save loss
        loss_meter.update(val=float(batch_loss), n=n)
        classification_loss_meter.update(val=float(classification_loss), n=n)
        regression_loss_meter.update(val=float(regression_loss), n=n)
        map_meter.update(val=mean_ap, n=n)

        # Free up GPU
        del classification_loss, regression_loss, batch_loss, results
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    return classification_loss_meter.avg, regression_loss_meter.avg, loss_meter.avg, map_meter.avg


def train(train_loader, model, optimizer, dx, device):
    model.train()

    return epoch_runner(train_loader, model, optimizer, dx, device)


def validate(val_loader, model, dx, device):
    model.eval()

    return epoch_runner(val_loader, model, None, dx, device)


def predict(test_image, model, device, boxes_to_view):
    label_map = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van',
                4: 'PS', 5: 'Truck', 6: 'Tram', 7: 'Misc'}
    color_map = {0: "red", 1: "green", 2: "black", 3: "cyan",
                  4: "blue", 5: "yellow", 6: "orange", 7: "purple"}
    
    test_image_tensor = torch.unsqueeze(test_image, 0)

    with torch.no_grad():
        test_image_tensor = test_image_tensor.float().to(device)
        features, classification, regression, anchors = model(test_image_tensor)

        results = retinanet_outputs(model, test_image_tensor, classification, regression, anchors)[0]
        for key in results:
            results[key] = results[key].cpu()
        predictedLabels, predictedBoxes = \
                    results["labels"], results["boxes"]
        
        # Plot the bounding boxes over the image
        labels, colors  = [], []
        for predicted_label in predictedLabels:
            labels.append(label_map[predicted_label.item()])
            colors.append(color_map[predicted_label.item()])
        boxes_to_view = min(boxes_to_view, len(labels))
        bounding_box_image = image_with_bounding_box(image=test_image,
                                                     boxes=predictedBoxes[:boxes_to_view],
                                                     labels=labels[:boxes_to_view],
                                                     colors=colors[:boxes_to_view])
    
    return bounding_box_image