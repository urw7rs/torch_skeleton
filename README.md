# torch_skeleton

Popular skeleton datasets and transforms

## Installation

```bash
pip install -e .
```

## Usage

```py
from torch_skeleton.datasetes import NTU, DiskCache, MapDataset
import torch_skeleton.transforms as T

dataset = NTU(
    root=".", num_classes=60, eval_type="subject",
    transform=T.Compose(
        [
            transforms.Denoise(),
            transforms.SubJoint(joint_id=1, all=False),
            transforms.SplitFrames(),
        ]
    ),
)

dataset = MapDataset(
    DiskCache( 
        root=dataset.root,
        dataset=dataset,
    ),
    T.Compose(
        [
            transforms.SampleFrames(num_frames=self.hparams.length),
            transforms.RandomRotate(degrees=self.theta),
            transforms.PadFrames(max_frames=self.hparams.length),
        ]
    )
)


x, y = dataset[0]
```
