import torch
from torch import nn

import einops

from skeleton.nn import DynamicsRepresentation, FrameLevelModule, SSHA, JointSemantics


class SSHASGN(nn.Module):
    def __init__(self, num_classes, length, num_joints, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.length = length
        self.num_joints = num_joints
        self.num_features = num_features

        self.dynamics_representation = DynamicsRepresentation(
            num_joints, [num_features, 64, 64]
        )

        self.ssha = SSHA(64, 64, num_node=num_joints, kernel_size=1, residual=False)

        """
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 64))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_joints, 64))

        self.transformer1 = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=64, batch_first=True
        )
        self.transformer2 = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=64, batch_first=True
        )
        self.transformer3 = nn.TransformerEncoderLayer(
            d_model=64, nhead=8, dim_feedforward=64, batch_first=True
        )
        """

        self.frame_level_module = FrameLevelModule(
            length, num_features=64, num_hidden=256, out_features=512
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        z = self.dynamics_representation(x)

        """
        z = einops.rearrange(z, "n t v c -> (n t) v c")

        z += self.pos_embedding
        cls_embedding = self.cls_embedding.repeat(z.size(0), 1, 1)
        z0 = torch.cat([cls_embedding, z], dim=1)

        z = self.transformer1(z0)
        z1 = z + z0
        z = self.transformer2(z1)
        z2 = z + z1
        z = self.transformer2(z2)
        z += z2

        z = einops.rearrange(z, "(n t) v c -> n t v c", n=x.size(0))

        # remove cls embedding
        z = z[:, :, 1:]
        """

        z = self.ssha(z)

        z = self.frame_level_module(z)

        logits = self.fc(z)

        return logits
