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
        splitDir = os.path.join(dataDir, split)
        self.imageDir = os.path.join(splitDir, "image_2")
        self.labelDir = None
        if split == "training":
            self.labelDir = os.path.join(splitDir, "label_2")
        self.dataset = self.load_paths()
        self.transform = transform

    def load_paths(self):
        dataPaths = []
        for i, imageName in enumerate(os.listdir(self.imageDir)):
            imageID = imageName.split(".")[0]
            imagePath = os.path.join(self.imageDir, f"{imageID}.png")
            if self.labelDir:
                labelPath = os.path.join(self.labelDir, f"{imageID}.txt")
            else:
                labelPath = None
            dataPaths.append((imagePath, labelPath))
            if i == 999:
                break

        return dataPaths
    
    def __getitem__(self, index:int):
        imageDir, labelDir = self.dataset[index]
        image = load_image(imageDir)
        image = self.transform(image)
        box, label = load_training_labels(labelDir)

        return image, torch.tensor(box), torch.tensor(label)
    
    def __len__(self):
        
        return len(self.dataset)