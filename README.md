# torch_skeleton

Popular skeleton datasets and transforms

## Insatllation

```bash
pip install -e .
```

## Usage

```py
from torch_skeleton.datasetes import NTUDataset
import torch_skeleton.transforms as T

dataset = NTUDataset(
    root=".", num_classes=60, eval_type="subject",
    transform=T.Compose(
        [
            transforms.SelectKBodies(k=2),
            transforms.SubJoint(joint_id=1, all=False),
            transforms.SplitFrames(),
            transforms.SampleFrames(num_frames=self.hparams.length),
            transforms.RandomRotate(degrees=self.theta),
            transforms.PadFrames(max_frames=self.hparams.length),
        ]
    ),
)

x, y = dataset[0]
```
