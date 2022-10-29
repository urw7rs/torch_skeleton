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
        split (str): split type, either ``"train"`` or ``"val"``
        transform (``Transform``): transform to apply to dataset
    """

    def __init__(
        self,
        root=".",
        split="train",
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

        paths = skel_utils.listdir(self.root, ext="json")
        self.file_paths = filter_split(paths, split)

    def __getitem__(self, index):
        path = self.file_paths[index]
        with open(path) as f:
            data = json.load(f)

        x = np.array(data["skeletons"]).astype(float)
        x = np.expand_dims(x, axis=0)
        y = int(data["label"]) - 1

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.file_paths)


def filter_split(paths, split):
    split_is_train = split == "train"

    split_paths = []
    for path in paths:
        # first two cameras for train, third camera for test
        camera_id = get_camera(path)
        in_train = camera_id != 3

        in_split = in_train == split_is_train

        if in_split:
            split_paths.append(path)

    return split_paths


def get_camera(path):
    file_name = osp.basename(path).split(".")[0]
    return int(file_name.split("_")[3][1:])
