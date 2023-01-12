import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST



class MNISTDataModule(pl.LightningDataModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitDataModule")
        parser.add_argument(
            "--train",
            type=str,
            default=os.getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        )
        parser.add_argument(
            "--test",
            type=str,
            default=os.getenv("SM_CHANNEL_TESTING", "/opt/ml/input/data/testing"),
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
        )
        parser.add_argument("--test_batch_size", type=int, default=100)

        parser.add_argument("--validation_fraction", type=float, default=0.1)

        parser.add_argument("--random_seed", type=float, default=1)

        return parent_parser

    def __init__(
        self,
        train: str = "./",
        test: str = "./",
        batch_size: int = 32,
        test_batch_size: int = 500,
        validation_fraction: float = 0.1,
        random_seed: int = 1,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            "batch_size",
            "validation_fraction",
            "train",
            "test",
            "test_batch_size",
            "random_seed",
        )

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.num_workers = num_workers

    def prepare_data(self):
        # download
        MNIST(
            self.hparams.train,  # type: ignore
            train=True,
            download=True,
        )
        MNIST(
            self.hparams.test,  # type: ignore
            train=False,
            download=True,
        )

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.hparams.train,  # type: ignore
                train=True,
                transform=self.transform,
            )
            data_length = len(mnist_full.data)
            validation_length = int(
                data_length * self.hparams.validation_fraction  # type: ignore
            )
            train_length = data_length - validation_length

            self.mnist_train, self.mnist_val = random_split(
                mnist_full,
                [train_length, validation_length],
                torch.Generator().manual_seed(self.hparams.random_seed),  # type: ignore
            )
            self.trainer.logger.log_hyperparams( #type: ignore
                {
                    "train_data_length": train_length,
                    "validation_data_length": validation_length,
                }
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(
                self.hparams.test,  # type: ignore
                train=False,
                transform=self.transform,
            )

            self.trainer.logger.log_hyperparams( #type: ignore
                {
                    "test_data_length": len(self.mnist_test.data),
                }
            )


    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.hparams.batch_size,  # type: ignore
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.hparams.test_batch_size,  # type: ignore
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.hparams.test_batch_size,  # type: ignore
            num_workers=self.num_workers,
        )
