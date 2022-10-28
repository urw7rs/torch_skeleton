from typing import Optional

from torch.utils.data import random_split

from pytorch_lightning import LightningDataModule

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from skeleton import transforms
from skeleton import filters
from skeleton.datasets import NTUDataset


class NTUDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        eval_batch_size: int = 64,
        num_classes: int = 60,
        eval_type: str = "subject",
        length: int = 20,
        theta: int = 17,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["data_dir", "batch_size", "eval_batch_size", "num_workers", "theta"]
        )

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.num_workers = num_workers

        self.theta = theta

    def setup(self, stage: Optional[str] = None):
        pre_transform = T.Compose(
            [
                transforms.SelectKBodies(k=2),
                transforms.SubJoint(joint_id=1, all=True),
                transforms.ParallelBone(first_id=0, second_id=1, axis=2),
                transforms.ParallelBone(first_id=4, second_id=8, axis=0),
                transforms.SplitFrames(),
            ]
        )
        if stage == "fit" or stage is None:
            self.train_set = NTUDataset(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="train",
                pre_filter=filters.filter_empty,
                pre_transform=pre_transform,
                transform=T.Compose(
                    [
                        transforms.SampleFrames(num_frames=self.hparams.length),
                        transforms.RandomRotate(degrees=self.theta),
                        transforms.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )

            """
            val_len = int(len(train_set) * 0.02)
            train_len = len(train_set) - val_len

            self.train_set, self.val_set = random_split(
                train_set, lengths=[train_len, val_len]
            )
            """
            self.val_set = NTUDataset(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="val",
                pre_filter=filters.filter_empty,
                pre_transform=pre_transform,
                transform=T.Compose(
                    [
                        transforms.SampleFrames(num_frames=self.hparams.length),
                        transforms.RandomRotate(degrees=self.theta),
                        transforms.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )
        else:
            self.test_set = NTUDataset(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="val",
                pre_filter=filters.filter_empty,
                pre_transform=pre_transform,
                transform=T.Compose(
                    [
                        transforms.SampleFrames(num_frames=self.hparams.length),
                        transforms.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
