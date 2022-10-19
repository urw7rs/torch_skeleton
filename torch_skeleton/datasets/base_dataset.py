import os
import os.path as osp
import multiprocessing as mp

import numpy as np

from typing import Optional

from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    @property
    def root_dir(self):
        return self.root

    @property
    def raw_dir(self):
        raw_dir = osp.join(self.root_dir, "raw")
        if not osp.exists(raw_dir):
            os.makedirs(raw_dir)
        return raw_dir

    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def raw_file_paths(self):
        return [osp.join(self.raw_dir, file_name) for file_name in self.raw_file_names]

    @property
    def download_paths(self):
        return ["./"]

    @property
    def parsed_dir(self):
        parsed_dir = osp.join(self.root_dir, "parsed")
        os.makedirs(parsed_dir, exist_ok=True)
        return parsed_dir

    def __init__(self, root: Optional[str] = None, num_workers: Optional[int] = None):
        super().__init__()

        self.root = "./" if root is None else root

        for path in self.download_paths:
            if not osp.exists(path):
                self.download(path)

        if num_workers == 0:
            use_multiprocessing = False
        elif num_workers is None:
            use_multiprocessing = True
        elif num_workers > 0:
            use_multiprocessing = True
        else:
            raise NotImplementedError

        if use_multiprocessing:
            with mp.Pool(num_workers) as p:
                self.parsed_file_paths = list(
                    p.imap_unordered(self._parse, self.raw_file_paths)
                )
        else:
            self.parsed_file_paths = list(map(self._parse, self.raw_file_paths))

    def download(self, path):
        return path

    def _parse(self, path):
        file_name = ".".join(osp.basename(path).split(".")[:-1])
        parsed_path = osp.join(self.parsed_dir, f"{file_name}.npy")

        if osp.exists(parsed_path):
            return parsed_path

        x = self.parse(path)
        with open(parsed_path, "wb") as f:
            np.save(f, x)

        return parsed_path

    def parse(self, path):
        return path

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.parsed_file_paths)
