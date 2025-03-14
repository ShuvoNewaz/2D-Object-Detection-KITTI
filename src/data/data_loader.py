import os
import torch
import numpy as np
from typing import List, Tuple
from PIL import Image
from torch.utils import data
from src.data.data_utils import *
# from src.training.utils import *


class ImageLoader(data.Dataset):
    def __init__(self, dataDir, split, transform):
        """
            args:
                dataDir: Directory of the dataset w.r.t. the root directory
                split: training or validation split
        """
        self.split = split

        # Validation set is extracted from training set
        split_map = {"training": "training", "validation": \
                     "training", "testing": "testing"}
        
        splitDir = os.path.join(dataDir, split_map[split])
        self.imageDir = os.path.join(splitDir, "image_2")
        self.labelDir = None
        if split in ["training", "validation"]:
            self.labelDir = os.path.join(splitDir, "label_2")
        self.dataset = self.load_paths()
        self.transform = transform

    def load_paths(self):
        image_dir_list = os.listdir(self.imageDir)[:200]
        number_of_images = len(image_dir_list)
        if self.split == "training":
            image_dir_list = image_dir_list[:int(7 / 10 * number_of_images)]
        elif self.split == "validation":
            image_dir_list = image_dir_list[int(7 / 10 * number_of_images):]
        dataPaths = []
        for i, imageName in enumerate(image_dir_list):
            imageID = imageName.split(".")[0]
            imagePath = os.path.join(self.imageDir, f"{imageID}.png")
            if self.labelDir:
                labelPath = os.path.join(self.labelDir, f"{imageID}.txt")
            else:
                labelPath = None
            dataPaths.append((imagePath, labelPath))

        return dataPaths
    
    def __getitem__(self, index:int):
        imageDir, labelDir = self.dataset[index]
        image = load_image(imageDir)
        image = self.transform(image)
        box, label = load_training_labels(labelDir)

        return image, torch.tensor(box), torch.tensor(label)
    
    def __len__(self):
        
        return len(self.dataset)