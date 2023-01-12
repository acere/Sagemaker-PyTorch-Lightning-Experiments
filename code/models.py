import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix


class MNISTModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        return parent_parser

    def __init__(self, hidden_dim: int = 128, learning_rate: float = 0.0001, **kwargs):
        super().__init__()
        self.save_hyperparameters("hidden_dim", "learning_rate")

        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.confmat = MulticlassConfusionMatrix(num_classes=10)

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim)  # type: ignore
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10)  # type: ignore

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = F.cross_entropy(probs, y)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        self.trainer.logger.log_epoch_duration(self.current_epoch)  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        loss = F.cross_entropy(probs, y)
        self.valid_acc.update(probs, y)

        self.log("validation_loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation_acc", self.valid_acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)

        self.test_acc.update(probs, y)
        self.confmat.update(probs, y)
        try:
            self.trainer.logger.log_confusion_matrix(  # type: ignore
                self.confmat.compute().tolist(),
                title="test-confusion-matrix",
            )
        except AttributeError:
            pass

        self.log("test_acc", self.test_acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
        )
