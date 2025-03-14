import torch


def kitti_collate_fn(batch):
    """
    returns:
        (x1, y1, x2, y2, label)
    """
    maxWidth, maxHeight, maxBoxNumber = 1248, 384, 0 # maxHeight hard-coded to avoid dimension mismatch after filter convolution
    
    for image, box, label in batch:
        if label is not None:
            maxBoxNumber = max(len(label), maxBoxNumber)

    # Pad the images, bounding boxes and labels if necessary
    padded_images = []
    padded_boxes = []
    padded_labels = []
    for image, box, label in batch:
        h, w = image.shape[1:]
        pad_h, pad_w = maxHeight - h, maxWidth - w
        padded_image = torch.cat([image, torch.zeros(3, pad_h, w)], dim=1)
        padded_image = torch.cat([padded_image, torch.zeros(3, maxHeight, pad_w)], dim=2)
        padded_images.append(padded_image)

        if label is not None:
            pad_size = maxBoxNumber - len(label)
            padded_boxes.append(torch.cat([box, torch.zeros(pad_size, 4)]))
            padded_labels.append(torch.cat([label, torch.tensor([-1] * pad_size)]))
        else:
            padded_boxes.append(None)
            padded_labels.append(None)

    # Stack boxes and labels to create a batch tensor
    batch_images = torch.stack(padded_images, dim=0)
    batch_boxes = torch.stack(padded_boxes, dim=0)
    batch_labels = torch.stack(padded_labels, dim=0)
    batch_boxes_labels = torch.concat([batch_boxes, torch.unsqueeze(batch_labels, dim=-1)], axis=-1)

    return batch_images, batch_boxes_labels