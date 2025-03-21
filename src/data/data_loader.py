import os
import torch
import numpy as np
from torch.utils import data
from src.data.data_utils import *
import matplotlib.pyplot as plt


class ImageLoader(data.Dataset):
    def __init__(self, dataDir, split, num_classes, transform):
        """
            args:
                dataDir: Directory of the dataset w.r.t. the root directory
                split: training or validation split
        """
        self.split = split
        self.num_classes = num_classes
        # Validation set is extracted from training set
        split_map = {"training": "training", \
                     "validation": "training", \
                        "testing": "testing"}
        
        splitDir = os.path.join(dataDir, split_map[split])
        self.imageDir = os.path.join(splitDir, "image_2")
        self.labelDir = None
        if split in ["training", "validation"]:
            self.labelDir = os.path.join(splitDir, "label_2")
        self.dataset = self.load_paths()
        self.transform = transform
        print(f"{split} data size: {len(self.dataset)} images.")

    def load_paths(self):
        image_dir_list = os.listdir(self.imageDir)#[:200]
        number_of_images = len(image_dir_list)
        if self.split == "training":
            image_dir_list = image_dir_list[:int(8 / 10 * number_of_images)]
        elif self.split == "validation":
            image_dir_list = image_dir_list[int(8 / 10 * number_of_images):]
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
        if self.split != "testing":
            box, label = load_training_labels(labelDir, self.num_classes)
            return image, torch.tensor(box), torch.tensor(label)
        else:
            return image, torch.tensor([[0, 0, 0, 0]]), torch.tensor([-1])
    
    def __len__(self):
        
        return len(self.dataset)
    
    def plot_label_distribution(self):
        label_map = {0: 'Car', 1: 'Pedestrian', 2: 'Cyclist', 3: 'Van',
                     4: 'Person_sitting', 5: 'Truck', 6: 'Tram', 7: 'Misc'}
        label_counter = {}
        for label_number in label_map:
            label_counter[label_map[label_number]] = 0

        for index in range(len(self.dataset)):
            labelDir = self.dataset[index][1]
            label = load_training_labels(labelDir, self.num_classes)[1]
            for single_box_label in label:
                if single_box_label != -1:
                    label_counter[label_map[single_box_label.item()]] += 1
        
        fig, ax = plt.subplots()
        fig.set_figheight(2)
        fig.set_figwidth(5)
        ax.bar(list(label_counter.keys()), list(label_counter.values()))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.title(f"{self.split} data label distribution")
        plt.show()