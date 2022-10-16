# torch_skeleton

Popular skeleton datasets and transforms

## Insatllation

```bash
pip install -e .
```

## Usage

```py
from torch_skeleton.datasetes import NTUDataset
from torch_skeleton import transforms

import torch_geometric.transforms as T

dataset = NTUDataset(
    root=".", num_classes=60, eval_type="subject",
    pre_filter=filters.filter_empty,
    pre_transform=T.Compose(
        [
            transforms.SelectKBodies(k=2),
            transforms.SubJoint(joint_id=1, all=False),
            transforms.SplitFrames(),
        ]
    )
    transform=T.Compose(
        [
            transforms.SampleFrames(num_frames=self.hparams.length),
            transforms.RandomRotate(degrees=self.theta),
            transforms.PadFrames(max_frames=self.hparams.length),
        ]
    ),
)

data = dataset[0]

x = data.x
y = data.y
```
