import os
from typing import Tuple, Union
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.data.data_loader import ImageLoader
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from src.training.train_utils import *
from src.data.collate import kitti_collate_fn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
        self,
        data_dir: str,
        model,
        optimizer: Optimizer,
        saved_model_dir: str,
        train_data_transforms: transforms.Compose,
        val_data_transforms: transforms.Compose,
        batch_size: int = 100,
        inp_size=(64, 64),
        load_from_disk: bool = True,
    ) -> None:
        self.saved_model_dir = saved_model_dir
        os.makedirs(saved_model_dir, exist_ok=True)
        self.model = model.to(device)
        # dataloader_args = {"num_workers": 1, "pin_memory": True} if device=="cuda" else {}
        dataloader_args = {}

        self.train_dataset = ImageLoader(
            data_dir, split="training", transform=train_data_transforms
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=kitti_collate_fn, **dataloader_args
        )

        # self.val_dataset = ImageLoader(
        #     data_dir, split="validation", transform=val_data_transforms
        # )
        # self.val_loader = DataLoader(
        #     self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
        # )

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_map_history = []
        self.validation_accuracy_history = []
        self.best_accuracy = 0

        # load the model from the disk if it exists
        if os.path.exists(saved_model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.saved_model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()


    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_classification_loss, train_regression_loss, train_loss, train_map = train(self.train_loader, self.model, self.optimizer, 0.05, device)

            self.train_loss_history.append(train_loss)
            self.train_map_history.append(train_map)
            
            # save_im = epoch_idx == (num_epochs - 1) # Save validation images and predictions at the last epoch
            # val_loss, val_acc = self.validate()
            # self.validation_loss_history.append(val_loss)
            # self.validation_accuracy_history.append(val_acc)
            # if val_acc > self.best_accuracy:
            #     self.best_accuracy = val_acc
            #     save_model(self.image_model, self.optimizer, self.model_dir)
                

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Classification Loss:{train_classification_loss:.4f}"
                + f" Train Regression Loss:{train_regression_loss:.4f}"
                + f" Train Total Loss:{train_loss:.4f}"
                # + f" Val Loss: {val_loss:.4f}"
                + f" Train Mean Average Precision: {train_map:.4f}"
                # + f" Validation Accuracy: {val_acc:.4f}"
            )

    
    def predict(self, saved_model):
        """Uses the best model on validation to predict the labels and store them"""
        indexTracker = 0
        for (x, y, imageDir, annDir) in self.val_loader:
            x = x.to(device)
            y = y.to(device)

            n = x.shape[0]
            logits = saved_model(x)
            # Return data to cpu
            x = x.cpu()
            y = y.cpu()

            # self.val_images[indexTracker:indexTracker+len(x)] = x.squeeze(1).numpy()
            self.true_labels[indexTracker:indexTracker+len(x)] = y.numpy()
            self.predictions[indexTracker:indexTracker+len(x)] = torch.argmax(logits, dim=1).cpu().numpy()
            self.valImageDir += list(imageDir)
            self.valAnnDir += list(annDir)
            indexTracker += len(x)
        self.model.cpu()

