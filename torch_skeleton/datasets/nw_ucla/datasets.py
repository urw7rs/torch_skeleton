import os.path as osp

from typing import Optional, Callable

from torch_skeleton.datasets.base_dataset import SkeletonDataset

import wget


class UCLA(SkeletonDataset):
    @property
    def root_dir(self):
        return osp.join(self.root, "NW-UCLA")

    @property
    def raw_file_paths(self):
        return super().raw_file_paths

    def __init__(
        self,
        split,
        root: Optional[str] = None,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_workers: int = 0,
    ):
        self.split = split

        self.urls = {
            "all_sqe.zip": "https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip"
        }

        self.checksums = {"all_sqe.zip": "6db59b046f5110fa644774afb2a906d2"}

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
        if osp.exists(path):
            return

        file_name = osp.basename(path)

        url = self.urls[file_name]
        wget.download(url, path)

    def parse(self, path):
        raise NotImplementedError

    def get(self, path, x):
        raise NotImplementedError
