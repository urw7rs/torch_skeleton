import pytest

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from torch_skeleton import datasets
import torch_skeleton.transforms as T

@pytest.mark.parametrize("num_classes", [60, 120])
@pytest.mark.parametrize("eval_type", ["subject", "camera", "setup"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_ntu_torch(root, num_classes, eval_type, split):
    dataset = datasets.NTU(
        root=root,
        num_classes=num_classes,
        eval_type=eval_type,
        split=split,
    )

    x, y = dataset[0]
    print(f"x size: {x.shape} y size: {y} {len(dataset)}")


@pytest.mark.parametrize("num_classes", [60, 120])
@pytest.mark.parametrize("eval_type", ["subject", "camera", "setup"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_ntu_torch_dataloaders(root, num_classes, eval_type, split, num_workers):
    dataset = datasets.NTU(
        num_classes=num_classes,
        eval_type=eval_type,
        split=split,
        root=root,
        transform=T.Compose(
            [
                T.SplitFrames(),
                T.SampleFrames(num_frames=20),
                T.PadFrames(max_frames=20),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=num_workers
    )

    x, y = next(iter(dataloader))


@pytest.mark.parametrize("split", ["train", "val"])
def test_ucla_torch(root, split):
    dataset = datasets.UCLA(root=root, split=split)

    x, y = dataset[0]
    print(f"x size: {x.shape} y size: {y} {len(dataset)}")


def test_ucla_torch_dataloaders(root, num_workers):
    dataset = datasets.UCLA(
        root=root,
        transform=T.Compose(
            [
                T.SplitFrames(),
                T.SampleFrames(num_frames=20),
                T.PadFrames(max_frames=20),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=num_workers
    )

    x, y = next(iter(dataloader))


class FakeData(Dataset):
    def __init__(self, length=100, max_bodies=1, min_frames=30, max_frames=300):
        super().__init__()

        self.length = length

        self.max_bodies = max_bodies
        self.min_frames = min_frames
        self.max_frames = max_frames

    def __getitem__(self, index):
        if self.max_bodies == 1:
            M = 1
        else:
            M = np.random.randint(1, self.max_bodies)

        if self.min_frames == self.max_frames:
            T = self.min_frames
        else:
            T = np.random.randint(self.min_frames, self.max_frames)

        x = np.random.randn(M, T, 17, 3)
        y = np.random.randint(0, 100)
        return x, y

    def __len__(self):
        return self.length


def test_disk_cache(root, num_workers):
    dataset = FakeData(length=100)
    dataset = datasets.DiskCache(root=root, dataset=dataset)

    x, y = dataset[0]


def test_disk_cache_dataloaders(root, num_workers):
    dataset = FakeData(length=100)
    dataset = datasets.DiskCache(
        root=root,
        dataset=dataset,
        transform=T.Compose(
            [
                T.SplitFrames(),
                T.SampleFrames(num_frames=20),
                T.PadFrames(max_frames=20),
            ]
        ),
    )

    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=num_workers
    )

    x, y = next(iter(dataloader))
