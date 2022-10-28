# torch_skeleton

Efficient datasets and transforms for skeleton data

## Installation

```bash
$ pip install torch_skeleton
```

## Datasets

Download and load raw dataset with preprocess

```python
from torch_skeleton.datasets import NTU
import torch_skeleton.transforms as T

# dwonload ntu skeleton dataset
ntu = NTU(
    root="data",
    num_classes=60,
    eval_type="subject",
    split="train",
    transform=T.Compose([
        T.Denoise(),
        T.CenterJoint(),
        T.SplitFrames(),
    ]),
)

x, y = ntu[0]
```

Cache preprocessed samples to disk

```python
from torch_skeleton.datasets import DiskCache

# cache preprocessing transforms to disk
cache = DiskCache(root="data/NTU", dataset=dataset)

x, y = cache[0]
```

Apply augmentations to a dataset

```python
from torch_skeleton.datasets import Apply

# cache preprocessing transforms to disk
cache = Apply(
    dataset=dataset, 
    transform=T.Compose([
        T.SampleFrames(num_frames=20),
        T.RandomRotate(degrees=17),
        T.PadFrames(max_frames=20),
    ]),
)

x, y = cache[0]
```

## License

`torch_skeleton` was created by Chanhyuk Jung. It is licensed under the terms
of the MIT license.
