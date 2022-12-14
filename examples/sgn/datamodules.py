from typing import Optional

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

from torch_skeleton.datasets import NTU, DiskCache, Apply

import torch_skeleton.transforms as T


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
        preprocess = T.Compose(
            [
                T.Denoise(),
                T.CenterJoint(joint_id=1, all=False),
                T.SplitFrames(),
            ]
        )

        if stage == "fit" or stage is None:
            dataset = NTU(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="train",
                transform=preprocess,
            )

            self.train_set = Apply(
                DiskCache(
                    root=dataset.root,
                    dataset=dataset,
                ),
                T.Compose(
                    [
                        T.SampleFrames(num_frames=self.hparams.length),
                        T.RandomRotate(degrees=self.theta),
                        T.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )

            dataset = NTU(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="val",
                transform=preprocess,
            )

            self.val_set = Apply(
                DiskCache(
                    root=dataset.root,
                    dataset=dataset,
                ),
                T.Compose(
                    [
                        T.SampleFrames(num_frames=self.hparams.length),
                        T.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )
        else:
            dataset = NTU(
                root=self.data_dir,
                num_classes=self.hparams.num_classes,
                eval_type=self.hparams.eval_type,
                split="val",
                transform=preprocess,
            )

            self.test_set = Apply(
                DiskCache(
                    root=dataset.root,
                    dataset=dataset,
                ),
                T.Compose(
                    [
                        T.SampleFrames(num_frames=self.hparams.length, num_clips=5),
                        T.PadFrames(max_frames=self.hparams.length),
                    ]
                ),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
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
