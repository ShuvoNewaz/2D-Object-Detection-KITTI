
"""
Contains functions with different data transforms
"""
import torch
from typing import Sequence, Tuple
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import random


def get_train_transforms(
    inp_size: Tuple[int, int]=(2, 2),
    pixel_mean: Sequence[float]=[0.485, 0.456, 0.406],
    pixel_std: Sequence[float]=[0.229, 0.224, 0.225]
) -> transforms.Compose:
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                            # transforms.Resize(inp_size),
                                            transforms.ColorJitter(),
                                            # transforms.RandomHorizontalFlip(0.5),
                                            transforms.Normalize(mean=pixel_mean, std=pixel_std)
                                        ])

    return train_transforms

def get_val_transforms(
    inp_size: Tuple[int, int]=(2, 2),
    pixel_mean: Sequence[float]=[0.485, 0.456, 0.406],
    pixel_std: Sequence[float]=[0.229, 0.224, 0.225]
) -> transforms.Compose:
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                            # transforms.Resize(inp_size),
                                            transforms.Normalize(mean=pixel_mean, std=pixel_std)
                                        ])

    return val_transforms


def get_test_transforms(
    inp_size: Tuple[int, int]=(2, 2),
    pixel_mean: Sequence[float]=[0.485, 0.456, 0.406],
    pixel_std: Sequence[float]=[0.229, 0.224, 0.225]
) -> transforms.Compose:
    test_transforms = transforms.Compose([transforms.ToTensor(),
                                        ])

    return test_transforms


def horizontal_flips(image, boxes, p):
    """
    Perform horizontal flip on an image tensor and adjust bounding boxes.
    args:
        image (torch.Tensor):   Image tensor of shape (C, H, W).
        boxes (torch.Tensor):   Bounding boxes (not handled in this version).
        p (float):              Probability of flipping.
    returns:
        torch.Tensor:       Flipped image.
    """
    c, h, w = image.shape
    affine_hflip_matrix = torch.tensor([[-1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0]]).unsqueeze(0)
    flipped = random.random() < p
    if flipped:
        # Flip image
        grid = F.affine_grid(affine_hflip_matrix,
                             [1, c, h, w], align_corners=True)
        image = F.grid_sample(image.unsqueeze(0), grid,
                                      mode='bilinear',
                                      padding_mode='zeros',
                                      align_corners=True).squeeze(0)
        
        # Adjust bounding boxes
        x_min = boxes[:, 0]
        y_min = boxes[:, 1]
        x_max = boxes[:, 2]
        y_max = boxes[:, 3]
        new_x_min = w - x_max
        new_x_max = w - x_min
        boxes = torch.stack([new_x_min, y_min, new_x_max, y_max], dim=1)
    return image, boxes