import os.path as osp

import shutil
import tempfile

import torch
from torch.utils.data import Dataset

import torch_skeleton.utils as skel_utils

from typing import Callable, Optional


class DiskCache(Dataset):
    """Cache ``Dataset`` instance to disk.

    Caches output of dataset to disk by creating a temporary directory at root.

    Args:
        root (str): root directory of cache
        dataset (``Dataset``): dataset to cache
    """

    def __init__(
        self,
        dataset: Dataset,
        root: str = ".",
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        skel_utils.makedirs(root)
        self.temp_dir = tempfile.TemporaryDirectory(dir=root)

        self.root = self.temp_dir.name
        self.transform = transform

        skel_utils.makedirs(self.root)
        shutil.rmtree(self.root)
        skel_utils.makedirs(self.root)

        self.dataset = dataset

    def cache_path(self, index):
        return osp.join(self.root, f"{index}.pt")

    def __getitem__(self, index):
        path = self.cache_path(index)

        if osp.exists(path):
            x, y = torch.load(path)
        else:
            x, y = self.dataset[index]

            torch.save([x, y], self.cache_path(index))

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        self.temp_dir.cleanup()


class Apply(Dataset):
    """Apply ``Transform`` to ``Dataset`` instance.

    Args:
        dataset (``Dataset``): dataset to apply transform to
        transform (``Transform``): transform to apply
    """

    def __init__(self, dataset: Dataset, transform: Callable):
        super().__init__()

        self.dataset = dataset

        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
