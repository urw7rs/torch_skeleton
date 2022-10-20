from torch import nn

from layers import DynamicsRepresentation, JointLevelModule, FrameLevelModule

import einops


class SGN(nn.Module):
    def __init__(self, num_classes, length, num_joints, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.length = length
        self.num_joints = num_joints
        self.num_features = num_features

        self.dynamics_representation = DynamicsRepresentation(
            num_joints, [num_features, 64, 64]
        )

        self.joint_level_module = JointLevelModule(num_joints)
        self.frame_level_module = FrameLevelModule(length)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = einops.rearrange(x, "n m t v c -> (n m) t v c")
        z = self.dynamics_representation(x)
        z = self.joint_level_module(z)
        z = self.frame_level_module(z)
        logits = self.fc(z)
        return logits
