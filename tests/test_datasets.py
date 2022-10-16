import pytest

import time
import multiprocessing as mp

from torch.utils.data import Dataset, DataLoader

from torch_skeleton import datasets
import torch_skeleton.transforms as T

import torchvision.transforms as TF


dataset = Dataset()


def load(i):
    global dataset

    return dataset[i]


@pytest.mark.parametrize("num_classes", [60, 120])
@pytest.mark.parametrize("eval_type", ["subject", "camera"])
@pytest.mark.parametrize("split", ["train", "val"])
def test_ntu_torch(root, num_classes, eval_type, split, num_workers):
    global dataset

    print(num_workers)

    dataset = datasets.NTUDataset(
        num_classes=num_classes,
        eval_type=eval_type,
        split=split,
        root=root,
        num_workers=num_workers,
        transform=TF.Compose(
            [
                T.SelectKBodies(k=2),
                T.SubJoint(joint_id=1, all=False),
                T.SplitFrames(),
                T.SampleFrames(num_frames=20),
                T.RandomRotate(degrees=17),
                T.PadFrames(max_frames=20),
            ]
        ),
    )

    x, y = dataset[0]
    print(f"x size: {x.shape} y size: {y} {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)

    t0 = time.time()
    for x, y in dataloader:
        pass
    t = time.time() - t0

    print(t)
