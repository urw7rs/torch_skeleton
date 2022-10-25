import os
import os.path as osp

import wget
import zipfile

import json

import numpy as np

from typing import Optional, Callable

from .base_dataset import LazyDataset


class UCLA(LazyDataset):
    urls = {"all_sqe.zip": "https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=1"}

    checksums = {"all_sqe.zip": "6db59b046f5110fa644774afb2a906d2"}

    @property
    def download_paths(self):
        return [osp.join(self.root, "raw", "all_sqe.zip")]

    def __init__(
        self,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = osp.join(root, "NW-UCLA")

        super().__init__()

        if transform is not None:
            self.transform = transform
        else:
            self.transform = lambda x: x

    def download(self, path):
        if not osp.exists(path):
            file_name = osp.basename(path)

            url = self.urls[file_name]
            wget.download(url, out=path)

            with zipfile.ZipFile(path, "r") as zip_ref:
                extract_dir = osp.dirname(path)
                zip_ref.extractall(extract_dir)

        paths = []
        with zipfile.ZipFile(path, "r") as zip_ref:
            extract_dir = osp.dirname(path)

            for path_obj in zip_ref.filelist:
                path = osp.join(extract_dir, path_obj.filename)
                if osp.isfile(path):
                    paths.append(path)

        return paths

    def open(self, path):
        data = np.load(path, allow_pickle=True)
        x = data["x"]
        y = data["y"]

        return self.transform(x), y

    def process(self, data):
        data = json.loads(data)

        x = np.array(data["skeletons"]).astype(float)
        y = data["label"]

        return (x, y)

    def save(self, data, path):
        x, y = data

        file_name = osp.basename(path)
        pre, _ = osp.splitext(file_name)

        path = osp.join(self.root, "processed", pre + ".npz")

        dir = osp.dirname(path)
        if not osp.exists(dir):
            os.makedirs(dir)

        np.savez(path, x=x, y=y)

        return path
