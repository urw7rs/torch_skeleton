import os.path as osp

import torch.multiprocessing as mp

from typing import List, Union, Tuple

import torch
from torch_geometric.data import Dataset, Data

from torch_skeleton import ntu
from torch_skeleton.utils import listdir


class NTUGraphDataset(Dataset):
    def __init__(
        self,
        root,
        num_classes=60,
        eval_type="subject",
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        assert num_classes in [60, 120], "only NTU-60 and NTU-120 exists"
        assert eval_type in ["subject", "camera"]

        self._num_classes = num_classes

        super().__init__(root, transform, pre_transform, pre_filter)

        data_list = torch.load(self.processed_paths[0])
        if num_classes == 120:
            data_list.extend(torch.load(self.processed_paths[1]))

        split_is_train = split == "train"

        def split_filter(data):
            in_train = (
                getattr(data, eval_type) in getattr(ntu, f"ntu_train_{eval_type}s")()
            )

            return in_train == split_is_train

        self.data_list = [data for data in data_list if split_filter(data)]

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def urls(self):
        urls = ["https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H"]
        if self.num_classes == 120:
            urls.append(
                "https://drive.google.com/uc?id=1tEbuaEqMxAV7dNc4fqu1O4M7mC6CJ50w"
            )

        return urls

    @property
    def checksums(self):
        checksums = ["67d9e24f858e5736a9826a2065e229fe"]
        if self.num_classes == 120:
            checksums.append("e8ae4bdd92c2be95dbd364ad54e82f89")

        return checksums

    @property
    def raw_dir(self) -> str:
        return osp.join(str(self.root), "NTU", "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(str(self.root), "NTU", "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        file_names = [
            "nturgbd_skeletons_s001_to_s017.zip",
        ]
        if self.num_classes == 120:
            file_names.append("nturgbd_skeletons_s018_to_s032.zip")

        return file_names

    @property
    def processed_file_names(self):
        file_names = ["nturgbd_skeletons_s001_to_s017.pt"]
        if self.num_classes == 120:
            file_names.append("nturgbd_skeletons_s018_to_s032.pt")
        return file_names

    def download(self):
        import gdown

        for url, raw_path, md5 in zip(self.urls, self.raw_paths, self.checksums):
            gdown.cached_download(
                url, path=raw_path, md5=md5, quiet=False, postprocess=gdown.extractall
            )

    def _process_file(self, path):
        with open(path) as f:
            string = f.read()
            skeleton_sequence = ntu.loads(string)

        x = ntu.as_numpy(skeleton_sequence)

        return x, path

    def _process_paths(self, paths):
        edge_index = torch.tensor(ntu.edge_index(), dtype=torch.long).t().contiguous()

        with mp.Pool() as p:
            data_list = []
            for x, path in p.imap_unordered(self._process_file, paths):
                x = torch.from_numpy(x)

                filename = ntu.get_filename(path)

                label = torch.tensor(ntu.label_from_name(filename), dtype=torch.long)

                setup = ntu.setup_from_name(filename)
                subject = ntu.subject_from_name(filename)
                camera = ntu.camera_from_name(filename)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=label,
                    setup=setup,
                    subject=subject,
                    camera=camera,
                )
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    def process(self):
        import resource

        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

        ntu60_paths = []
        ntu120_paths = []
        for path in listdir(self.raw_dir, ext=".skeleton"):
            filename = ntu.get_filename(path)
            setup = ntu.setup_from_name(filename)

            if setup > 17:
                ntu120_paths.append(path)
            else:
                ntu60_paths.append(path)

        if not osp.exists(self.processed_paths[0]):
            data_list = self._process_paths(ntu60_paths)
            torch.save(data_list, self.processed_paths[0])

        if self._num_classes == 120:
            data_list = self._process_paths(ntu120_paths)
            torch.save(data_list, self.processed_paths[1])

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
