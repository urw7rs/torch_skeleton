import os
import os.path as osp

import gdown
import urllib.request

from typing import Optional, Callable

import zipfile
import numpy as np

from . import ntu
from . import utils

from ..base_dataset import CachingDataset
from torch_skeleton.utils import check_md5sum


class NTUDataset(CachingDataset):
    urls = {
        "nturgbd_skeletons_s001_to_s017.zip": "https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H",
        "nturgbd_skeletons_s018_to_s032.zip": "https://drive.google.com/uc?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w",
        "NTU_RGBD120_samples_with_missing_skeletons.txt": "https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD120_samples_with_missing_skeletons.txt",
        "NTU_RGBD_samples_with_missing_skeletons.txt": "https://raw.githubusercontent.com/shahroudy/NTURGB-D/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt",
    }

    checksums = {
        "nturgbd_skeletons_s001_to_s017.zip": "67d9e24f858e5736a9826a2065e229fe",
        "nturgbd_skeletons_s018_to_s032.zip": "e8ae4bdd92c2be95dbd364ad54e82f89",
    }

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

        return [osp.join(self.root, "raw", file_name) for file_name in file_names]

    def __init__(
        self,
        num_classes,
        eval_type,
        split,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = osp.join(root, "NTU")

        self.num_classes = num_classes
        self.eval_type = eval_type
        self.split = split

        super().__init__()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x

    def download(self, path):
        file_name = osp.basename(path)

        url = self.urls[file_name]

        if not osp.exists(path):
            if file_name.split(".")[-1] == "txt":
                urllib.request.urlretrieve(url, path)
            else:
                gdown.download(url, output=path, quiet=False)

                if check_md5sum(path, self.checksums[file_name]) is False:
                    print("Warning! md5sum doesn't match")

                with zipfile.ZipFile(path, "r") as zip_ref:
                    if file_name == "nturgbd_skeletons_s001_to_s017.zip":
                        extract_dir = osp.dirname(path)

                    elif file_name == "nturgbd_skeletons_s018_to_s032.zip":
                        extract_dir = osp.join(osp.dirname(path), "nturgb+d_skeletons")
                    zip_ref.extractall(extract_dir)

        if file_name.split(".")[-1] == "txt":
            return []

        paths = []
        with zipfile.ZipFile(path, "r") as zip_ref:
            if file_name == "nturgbd_skeletons_s001_to_s017.zip":
                extract_dir = osp.dirname(path)

            elif file_name == "nturgbd_skeletons_s018_to_s032.zip":
                extract_dir = osp.join(osp.dirname(path), "nturgb+d_skeletons")

            for path_obj in zip_ref.filelist:
                path = osp.join(extract_dir, path_obj.filename)
                if osp.isfile(path):
                    paths.append(path)

        for path in self.download_paths:
            if path.split(".")[-1] == "txt":
                with open(path) as f:
                    # skip three lines
                    missing_files = f.read().splitlines()[3:]

                paths = filter_missing(paths, missing_files)

        paths = filter_num_classes(paths, self.num_classes)
        paths = filter_split(paths, self.eval_type, self.split)

        return paths

    def open(self, path):
        x = np.load(path)
        y = utils.label_from_name(osp.basename(path))
        return self.transform(x), y

    def process(self, data):
        skeleton_sequence = ntu.loads(data)
        x = ntu.as_numpy(skeleton_sequence)
        return x

    def save(self, data, path):
        file_name = osp.basename(path)
        pre, _ = osp.splitext(file_name)

        path = osp.join(self.root, "processed", pre + ".npy")

        dir = osp.dirname(path)
        if not osp.exists(dir):
            os.makedirs(dir)

        np.save(path, data)

        return path


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
