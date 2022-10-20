from typing import List

from einops import parse_shape, rearrange, reduce, repeat
from einops.layers.torch import Rearrange

import torch
from torch import nn

from torch_geometric.nn import Sequential

import math


class DynamicsRepresentation(nn.Module):
    def __init__(self, num_joints, dims=[3, 64, 64]):
        super().__init__()

        num_features = dims[0]

        self.batch_norm1 = nn.BatchNorm1d(num_features * num_joints)
        self.batch_norm2 = nn.BatchNorm1d(num_features * num_joints)

        self.embed_pos = Embed(dims)
        self.embed_vel = Embed(dims)

    def forward(self, joint):
        zero = torch.zeros_like(joint[:, 0:1])
        padded_position = torch.cat([zero, joint], dim=1)
        velocity = padded_position[:, 1:] - padded_position[:, :-1]

        shape = parse_shape(joint, "n t v _")

        joint = rearrange(joint, "n t v c -> n (v c) t")
        joint = self.batch_norm1(joint)
        joint = rearrange(joint, "n (v c) t -> n t v c", **shape)
        pos_embedding = self.embed_pos(joint)

        velocity = rearrange(velocity, "n t v c -> n (v c) t")
        velocity = self.batch_norm2(velocity)
        velocity = rearrange(velocity, "n (v c) t -> n t v c", **shape)
        vel_embedding = self.embed_vel(velocity)

        fused_embedding = pos_embedding + vel_embedding

        return fused_embedding


class JointLevelModule(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        one_hot_joint = torch.eye(num_joints)
        self.register_buffer("j", one_hot_joint)

        self.embed_joint = Embed([num_joints, 64, 64])

        self.compute_adjacency_matrix = AdjacencyMatrix(128, 256)

        self.gcn1 = self.build_gcn(128, 128)
        self.gcn2 = self.build_gcn(128, 256)
        self.gcn3 = self.build_gcn(256, 256)

    def forward(self, z):
        N, T, V, C = z.size()

        j = repeat(self.embed_joint(self.j), "v1 v2 -> n t v1 v2", n=N, t=T)
        z = torch.cat([z, j], dim=-1)

        # G: N, T, V, V
        G = self.compute_adjacency_matrix(z)

        # x: N, T, V, C, adj: N, T, V, V
        x = self.gcn1(z, G)
        x = self.gcn2(x, G)
        x = self.gcn3(x, G)

        return x

    def build_gcn(self, in_channels, out_channels):
        return Sequential(
            "x, adj",
            [
                (
                    ResidualGCN(in_channels, out_channels),
                    "x, adj -> x",
                ),
                Rearrange("n t v c -> n c v t"),
                nn.BatchNorm2d(out_channels),
                Rearrange("n c v t -> n t v c"),
                nn.ReLU(),
            ],
        )


class FrameLevelModule(nn.Module):
    def __init__(self, length, num_features=256, num_hidden=256, out_features=512):
        super().__init__()

        one_hot_frame = torch.eye(length)
        self.register_buffer("f", one_hot_frame)

        self.embed_frame = Embed([length, num_features // 4, num_features])

        self.tcn1 = nn.Sequential(
            nn.Conv1d(num_features, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
        )

        self.drop_out = nn.Dropout1d(0.2)

        self.tcn2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_features, kernel_size=1),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

        self.apply(kaiming_init)

    def forward(self, z):
        N, T, V, C = z.size()
        z = z + self.embed_frame(self.f)[None, :, None]
        z = reduce(z, "n t v c -> n t c", "max")  # spatial max pooling

        x = rearrange(z, "n t c -> n c t")
        x = self.tcn1(x)
        x = self.drop_out(x)
        x = self.tcn2(x)

        x = reduce(x, "n c t -> n c", "max")  # temporal max pooling

        return x


class Embed(nn.Module):
    def __init__(self, feature_dims: List[int] = [64, 64, 64]):
        super().__init__()

        modules = []
        for in_channels, out_channels in zip(feature_dims[:-1], feature_dims[1:]):
            module = self.build_fc(in_channels, out_channels)
            modules.append(module)

        self.fc = nn.Sequential(*modules)
        self.apply(kaiming_init)

    def forward(self, x):
        return self.fc(x)

    def build_fc(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        n = m.weight.shape[0] * m.weight.shape[1]
        nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / n))

    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / n))


class AdjacencyMatrix(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.theta = nn.Linear(in_features, out_features)
        self.phi = nn.Linear(in_features, out_features)

        self.apply(kaiming_init)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z):
        theta = self.theta(z)
        phi = self.phi(z)

        phi = rearrange(phi, "n t v c -> n t c v")
        S = torch.matmul(theta, phi)
        # S, n v v
        G = self.softmax(S)
        # G, n v v
        return G


class ResidualGCN(nn.Module):
    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved

        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.zeros_(self.lin1.weight)

        self.lin2 = nn.Linear(in_channels, out_channels, bias=True)
        nn.init.kaiming_normal_(self.lin2.weight, nonlinearity="relu")

    def forward(self, x, adj):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        agg = torch.matmul(adj, x)

        out = self.lin1(agg) + self.lin2(x)

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels})"
