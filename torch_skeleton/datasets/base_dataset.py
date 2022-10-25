from multiprocessing import process
import os
import os.path as osp

from torch.utils.data import Dataset


class CachingDataset(Dataset):
    @property
    def download_paths(self):
        return []

    @property
    def raw_file_paths(self):
        return self._raw_file_paths

    def __init__(self):
        super().__init__()

        raw_file_paths = []
        for path in self.download_paths:
            dir = osp.dirname(path)
            if not osp.exists(dir):
                os.makedirs(dir, exist_ok=True)

            raw_file_paths.extend(self.download(path))

        self._raw_file_paths = raw_file_paths

        self._path_map = {}

    def download(self, path):
        return []

    def open_raw(self, path):
        with open(path, encoding="utf-8") as f:
            string = f.read()
        return string

    def open(self, path):
        raise NotImplementedError

    def process(self, data):
        raise NotImplementedError

    def save(self, data, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        path = self.raw_file_paths[idx]
        processed_path = self._path_map.get(path, None)

        if processed_path is None:
            data = self.open_raw(path)
            data = self.process(data)

            processed_path = self.save(data, path)

            assert osp.exists(processed_path)

            self._path_map[path] = processed_path

        return self.open(processed_path)

    def __len__(self):
        return len(self.raw_file_paths)
