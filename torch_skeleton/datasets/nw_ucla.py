import os.path as osp

import json
import numpy as np

from torch.utils.data import Dataset

import torch_skeleton.utils as skel_utils

from typing import Callable, Optional


class UCLA(Dataset):
    """`NW-UCLA <http://wangjiangb.github.io/my_data.html>`_ Dataset.

    Args:
        root (str): root directory of dataset
        transform (``Transform``): transform to apply to dataset
    """

    def __init__(
        self,
        root=".",
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        self.root = osp.join(root, "NW-UCLA")
        self.transform = transform

        path = osp.join(self.root, "all_sqe.zip")
        if not skel_utils.downloaded(path):
            skel_utils.download_url(
                "https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=1", path
            )

            skel_utils.extract_zip(path, self.root)

        self.file_paths = skel_utils.listdir(self.root, ext="json")

    def __getitem__(self, index):
        path = self.file_paths[index]
        with open(path) as f:
            data = json.load(f)

        x = np.array(data["skeletons"]).astype(float)
        x = np.expand_dims(x, axis=0)
        y = data["label"]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.file_paths)
