import os.path as osp

import torch.multiprocessing as mp

from torch.utils.data import Dataset

from typing import Any, Callable, List, Optional, Tuple, Union

import zipfile

import einops

from torch_skeleton import ntu
from torch_skeleton.utils import listdir


class NTUTorchDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__()

        root_dir = osp.join(root, "NTU")

        if download:
            import gdown

            urls = {
                "nturgbd_skeletons_s001_to_s017.zip": "https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H",
                "nturgbd_skeletons_s018_to_s032.zip": "https://drive.google.com/uc?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w",
            }

            checksums = {
                "nturgbd_skeletons_s001_to_s017.zip": "67d9e24f858e5736a9826a2065e229fe",
                "nturgbd_skeletons_s018_to_s032.zip": "e8ae4bdd92c2be95dbd364ad54e82f89",
            }

            zip_names = [
                "nturgbd_skeletons_s001_to_s017.zip",
                "nturgbd_skeletons_s018_to_s032.zip",
            ]

            for zip_name in zip_names:
                url = urls[zip_name]
                md5 = checksums[zip_name]

                gdown.cached_download(
                    url, path=osp.join(root_dir, zip_name), md5=md5, quiet=False
                )

            for zip_path in zip_names:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(root_dir)

        self.path_list = listdir(root_dir, ext="skeleton")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x

        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = lambda x: x

    def __getitem__(self, idx):
        path = self.path_list[idx]

        with open(path) as f:
            skeleton_sequence = ntu.loads(f.read())
            x = ntu.as_numpy(skeleton_sequence)
            x = einops.rearrange(x, "m t v c -> c v t m")

        filename = ntu.get_filename(path)
        y = ntu.label_from_name(filename)

        x = self.transform(x)
        y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.path_list)
