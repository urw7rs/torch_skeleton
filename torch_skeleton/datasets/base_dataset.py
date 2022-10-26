import os.path as osp

import shutil
import tempfile

import torch
from torch.utils.data import Dataset

import torch_skeleton.utils as skel_utils

from typing import Callable, Optional


class DiskCache(Dataset):
    def __init__(self, root, dataset: Dataset, transform: Optional[Callable] = None):
        super().__init__()

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


class MapDataset(Dataset):
    def __init__(self, dataset: Dataset, fn: Callable):
        super().__init__()

        self.dataset = dataset

        self.fn = fn

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = self.fn(x)
        return x, y

    def __len__(self):
        return len(self.dataset)
