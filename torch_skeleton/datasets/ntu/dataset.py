import os.path as osp

import gdown
import urllib.request

import numpy as np

from typing import Optional, Callable

import zipfile

from . import ntu
from . import utils

from torch_skeleton.datasets.base_dataset import SkeletonDataset
from torch_skeleton.utils import listdir


class NTUDataset(SkeletonDataset):
    @property
    def root_dir(self):
        return osp.join(self.root, "NTU")

    @property
    def download_paths(self):
        if self.num_classes == 60:
            file_names = [
                "nturgbd_skeletons_s001_to_s017.zip",
                "NTU_RGBD_samples_with_missing_skeletons.txt",
            ]
        elif self.num_classes == 120:
            file_names = [
                "nturgbd_skeletons_s001_to_s017.zip",
                "nturgbd_skeletons_s018_to_s032.zip",
                "NTU_RGBD120_samples_with_missing_skeletons.txt",
            ]
        else:
            raise NotImplementedError

        return [osp.join(self.raw_dir, file_name) for file_name in file_names]

    @property
    def raw_file_paths(self):
        paths = listdir(self.raw_dir, ext="skeleton")

        for path in self.download_paths:
            if path.split(".")[-1] == "txt":
                with open(path) as f:
                    missing_files = f.read().splitlines()[3:]

        paths = filter_missing(paths, missing_files)
        paths = filter_num_classes(paths, self.num_classes)
        paths = filter_split(paths, self.eval_type, self.split)
        return paths

    def __init__(
        self,
        num_classes,
        eval_type,
        split,
        root: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_workers: int = 0,
    ):
        self.num_classes = num_classes
        self.eval_type = eval_type
        self.split = split

        self.urls = {
            "nturgbd_skeletons_s001_to_s017.zip": "https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H",
            "nturgbd_skeletons_s018_to_s032.zip": "https://drive.google.com/uc?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w",
            "NTU_RGBD120_samples_with_missing_skeletons.txt": "https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt",
            "NTU_RGBD_samples_with_missing_skeletons.txt": "https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt",
        }

        self.checksums = {
            "nturgbd_skeletons_s001_to_s017.zip": "67d9e24f858e5736a9826a2065e229fe",
            "nturgbd_skeletons_s018_to_s032.zip": "e8ae4bdd92c2be95dbd364ad54e82f89",
        }

        super().__init__(root=root, preprocess=preprocess, num_workers=num_workers)

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x

        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = lambda x: x

    def download(self, path):
        file_name = osp.basename(path)

        url = self.urls[file_name]

        if file_name == "nturgbd_skeletons_s001_to_s017.zip":
            md5 = self.checksums[file_name]
            gdown.cached_download(url, path=path, md5=md5, quiet=False)

            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(osp.dirname(path))
        elif file_name == "nturgbd_skeletons_s018_to_s032.zip":
            md5 = self.checksums[file_name]
            gdown.cached_download(url, path=path, md5=md5, quiet=False)

            with zipfile.ZipFile(path, "r") as zip_ref:
                zip_ref.extractall(osp.join(osp.dirname(path), "nturgb+d_skeletons"))
        else:
            urllib.request.urlretrieve(url, path)

    def parse(self, path):
        with open(path, encoding="utf-8") as f:
            skeleton_sequence = ntu.loads(f.read())
            x = ntu.as_numpy(skeleton_sequence)
        return x

    def __getitem__(self, idx):
        path = self.load_file_paths[idx]

        with open(path, "rb") as f:
            x = np.load(f)

        y = utils.label_from_name(osp.basename(path))

        x = self.transform(x)
        y = self.target_transform(y)

        return x, y

    def __len__(self):
        return len(self.parsed_file_paths)


def filter_num_classes(paths, num_classes):
    ntu60_paths = []
    ntu120_paths = []
    for path in paths:
        setup = utils.setup_from_name(osp.basename(path))

        if setup > 17:
            ntu120_paths.append(path)
        else:
            ntu60_paths.append(path)

    if num_classes == 60:
        return ntu60_paths
    elif num_classes == 120:
        return ntu120_paths
    else:
        raise NotImplementedError


def filter_split(paths, eval_type, split):
    split_is_train = split == "train"

    get_eval = getattr(utils, f"{eval_type}_from_name")
    train_evals = getattr(utils, f"ntu_train_{eval_type}s")()

    split_paths = []
    for path in paths:
        in_train = get_eval(osp.basename(path)) in train_evals

        in_split = in_train == split_is_train

        if in_split:
            split_paths.append(path)

    return split_paths


def filter_missing(paths, missing_files):
    filtered_paths = []
    for path in paths:
        file_name = osp.basename(path).split(".")[0]
        if file_name not in missing_files:
            filtered_paths.append(path)
    return filtered_paths
