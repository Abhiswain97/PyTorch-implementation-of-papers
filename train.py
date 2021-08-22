from AlexNet.model import AlexNet
import torchvision
from torch.utils.data import DataLoader, dataset, random_split
import torch.optim as optimizer
import torch.nn as nn
import torch

import pytorch_lightning as pl

import numpy as np


class config:
    BATCH_SIZE = 32


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AlexNet()
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optimizer.SGD(self.model.parameters(), lr=1e-5)

    def prepare_data(self, valid_size=0.2, random_seed=42, shuffle=True):
        """Can be done in __init__() method also"""
        transforms = {
            "train": torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            "valid": torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        }

        self.train_dataset = torchvision.datasets.MNIST(root='.',
                                                        train=True, download=True, transform=transforms["train"]
                                                        )
        self.valid_dataset = torchvision.datasets.MNIST(root='.',
                                                        train=False, download=True, transform=transforms["valid"]
                                                        )

    def train_dataloader(self):
        """REQUIRED"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
        )

    def val_dataloader(self):
        """REQUIRED"""
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=1,
        )

    def forward(self, x):
        """REQUIRED"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """REQUIRED"""
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss(preds, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        """REQUIRED"""
        images, labels = batch
        preds = self.forward(images)
        loss = self.loss(preds, labels)

        return loss

    def validation_epoch_end(self, outputs):
        """(OPTIONAL) To compute statistics"""
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        print(f"Validation Loss: {avg_loss}")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(
                {
                    "best_loss": avg_loss,
                    "model": self.model,
                    "model_state_dict": self.model.state_dict(),
                },
                "best_model.pt",
            )

        return {"val_loss": avg_loss}


if __name__ == "__main__":
    model = LitModel()

    trainer = pl.Trainer(gpus=1, max_epochs=10)

    trainer.fit(model=model)
