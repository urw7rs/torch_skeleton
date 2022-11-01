import os.path as osp
import pickle

import numpy as np

from torch.utils.data import Dataset

import torch_skeleton.utils as skel_utils

from typing import Callable, Optional


class BABEL(Dataset):
    """`BABEL <https://babel.is.tue.mpg.de/index.html>`_ Dataset

    Downloads pre-processed datasets

    Args:
        root (str): root directory of dataset
        num_classes (int): number of classes, ``60`` for BABEL60, ``120`` for BABEL120
        extra (bool): flag to use extra data
        split (str): split type. either ``"train"`` or ``"val"`` or ``"test"``
        transform (``Transform``): transform to apply to dataset
    """

    def __init__(
        self,
        root: str = ".",
        num_classes: int = 60,
        split: str = "train",
        extra: bool = False,
        transform: Optional[Callable] = None,
    ):
        super().__init__()

        if extra:
            assert split != "test", "test set is not available"

        self.root = osp.join(root, "BABEL")
        self.transform = transform

        if extra:
            file_name = "babel_dense_and_extra_feats_labels.tar.gz"
            url = f"https://human-movement.is.tue.mpg.de/{file_name}"
        else:
            file_name = "babel_feats_labels.tar.gz"
            url = f"https://human-movement.is.tue.mpg.de/{file_name}"

        path = osp.join(self.root, file_name)
        if not skel_utils.downloaded(path):
            skel_utils.download_url(url, path=path)
            skel_utils.extract_tar(path, self.root)

        if extra:
            babel_dir = "babel_extra_feats_labels"
        else:
            babel_dir = "release"

        root_dir = osp.join(self.root, babel_dir)

        extra_str = "extra_" if extra else ""
        data_path = osp.join(root_dir, f"{split}_{extra_str}ntu_sk_{num_classes}.npy")
        label_path = osp.join(root_dir, f"{split}_{extra_str}label_{num_classes}.pkl")

        X = np.load(data_path)  # N C T V M
        self.X = np.transpose(X, axes=(0, 4, 2, 3, 1))

        with open(label_path, "rb") as f:
            seg_id, annotations = pickle.load(f, encoding="latin1")

        label, sid, chunk_n, anntr_id = annotations

        self.metadata = {
            "seg_id": seg_id,
            "sid": sid,
            "chunk_n": chunk_n,
            "anntr_id": anntr_id,
        }

        self.Y = label

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.Y)
