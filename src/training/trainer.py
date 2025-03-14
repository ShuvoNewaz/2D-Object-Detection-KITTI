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

        self.val_dataset = ImageLoader(
            data_dir, split="validation", transform=val_data_transforms
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=True, collate_fn=kitti_collate_fn, **dataloader_args
        )

        self.optimizer = optimizer

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_map_history = []
        self.validation_map_history = []
        self.best_map = 0

        # load the model from the disk if it exists
        if os.path.exists(saved_model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.saved_model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()


    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):

            train_classification_loss, train_regression_loss, train_loss, train_map = \
                train(self.train_loader, self.model, self.optimizer, 0.05, device)
            self.train_loss_history.append(train_loss)
            self.train_map_history.append(train_map)
            
            val_classification_loss, val_regression_loss, val_loss, val_map = \
                validate(self.val_loader, self.model, None, 0.05, device)
            self.validation_loss_history.append(val_loss)
            self.validation_map_history.append(val_map)

            if val_map > self.best_map:
                self.best_map = val_map
                save_model(self.model, self.optimizer, self.saved_model_dir)
                
            print(f"Epoch {epoch_idx + 1}:")
            # print(f"\tTrain Classification Loss:{train_classification_loss:.4f}")
            # print(f"\tTrain Regression Loss:{train_regression_loss:.4f}")
            print(f"\tTrain Total Loss:{train_loss:.4f}")
            print(f"\tTrain Mean Average Precision: {train_map:.4f}")
            print("")
            # print(f"\tValidation Classification Loss:{val_classification_loss:.4f}")
            # print(f"\tValidation Regression Loss:{val_regression_loss:.4f}")
            print(f"\tValidation Total Loss:{val_loss:.4f}")
            print(f"\tValidation Mean Average Precision: {val_map:.4f}")