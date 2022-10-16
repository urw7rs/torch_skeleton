import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics import Accuracy

from models import SGN


class LitSGN(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        length=20,
        num_joints=25,
        num_features=3,
        lr=0.001,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.sgn = SGN(num_classes, length, num_joints, num_features)

        self.metrics = nn.ModuleDict(
            {
                "train_acc": Accuracy(),
                "val_acc": Accuracy(),
                "test_acc": Accuracy(),
            }
        )

    def training_step(self, batch, batch_idx):
        x, y = batch

        batch_size = y.size(0)

        logits = self.sgn(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.metrics["train_acc"](preds, y.int())
        self.log(
            "train_acc",
            self.metrics["train_acc"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        batch_size = y.size(0)

        logits = self.sgn(x)
        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        self.metrics["val_acc"](preds, y.int())
        self.log(
            "val_acc",
            self.metrics["val_acc"],
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        batch_size = y.size(0)
        num_clips = x.size(1)

        x = torch.flatten(x, 0, 1)
        logits = self.sgn(x)

        logits = logits.view(batch_size, num_clips, -1)
        logits = logits.mean(1)

        loss = F.cross_entropy(logits, y, label_smoothing=0.1)

        preds = logits.argmax(dim=1)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.metrics["test_acc"](preds, y.int())
        self.log(
            "test_acc",
            self.metrics["test_acc"],
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler_dict = {
            "scheduler": MultiStepLR(optimizer, milestones=[60, 90, 110]),
            "interval": "epoch",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }

    def on_fit_start(self) -> None:
        from fvcore.nn import FlopCountAnalysis, flop_count_table

        x, y = next(iter(self.trainer.datamodule.train_dataloader()))

        x = x.to(self.device)
        x = x[:1]

        flops = FlopCountAnalysis(self.sgn, x)
        print(flop_count_table(flops, max_depth=10))
