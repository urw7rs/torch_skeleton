import pytest

import numpy as np

from torch_skeleton.datasets import NTUDataset
from torch_skeleton.transforms import functions


def test_denoising(root, num_workers):
    sample = NTUDataset(
        num_classes=60,
        eval_type="camera",
        split="train",
        root=root,
        num_workers=num_workers,
    )[272]

    x, y = sample

    x = functions.get_raw_denoised_data(x)
    print(x.shape)
    return

    num_bodies = x.shape[0]

    actors = np.split(x, indices_or_sections=num_bodies, axis=0)
    print(actors[0].shape, actors[1].shape, actors[2].shape, num_bodies)

    actor1 = x[0:1]
    actor2 = x[1:2]

    _, indices = functions.get_indices(actor1)
    print(indices.shape)

    intersect = functions.intersect_indices(actor1, actor2)
    print(len(intersect))

    actor2 = np.zeros_like(actor1)

    intersect = functions.intersect_indices(actor1, actor2)
    print(len(intersect))

    print(x.shape)
    x = functions.get_two_actors_points(x)
    print(x.shape)
