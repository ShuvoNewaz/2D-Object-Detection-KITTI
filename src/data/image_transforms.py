
"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple

import numpy as np
import torchvision.transforms as transforms

def get_train_transforms(
    inp_size: Tuple[int, int]=(2, 2), pixel_mean: Sequence[float]=[0.485, 0.456, 0.406], pixel_std: Sequence[float]=[0.229, 0.224, 0.225]
) -> transforms.Compose:
    train_transforms = transforms.Compose([transforms.ToTensor(),
                                            # transforms.Resize(inp_size),
                                            # transforms.ColorJitter(),
                                            # transforms.RandomHorizontalFlip(0.5),
                                            # transforms.Normalize(mean=pixel_mean, std=pixel_std)
                                        ])

    return train_transforms

def get_val_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    val_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(inp_size),
                                            transforms.Normalize(mean=pixel_mean, std=pixel_std)
                                        ])

    return val_transforms