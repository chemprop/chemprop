from abc import abstractmethod

import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from chemprop.data.collate import BatchMolGraph


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike, pad: int = 0):
        super().__init__()

        if pad > 0:
            mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
            scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))

    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler):
        return cls(scaler.mean_, scaler.scale_)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass


class ScaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return (X - self.mean) / self.scale


class UnscaleTransform(_ScaleTransformMixin):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return X * self.scale + self.mean


class GraphTransform(nn.Module):
    def __init__(self, V_mean: ArrayLike, V_scale: ArrayLike, atom_fdim: int, E_mean: ArrayLike, E_scale: ArrayLike, bond_fdim: int):
        super().__init__()

        self.V_transform = ScaleTransform(V_mean, V_scale, pad=atom_fdim)
        self.E_transform = ScaleTransform(E_mean, E_scale, pad=bond_fdim)

    def from_standard_scaler(V_scaler: StandardScaler, E_scaler: StandardScaler):
        return GraphTransform(V_scaler.mean_, V_scaler.scale_, E_scaler.mean_, E_scaler.scale_)

    def forward(self, bmg: BatchMolGraph) -> BatchMolGraph:
        if self.training:
            return bmg

        bmg.V = self.V_transform(bmg.V)
        bmg.E = self.E_transform(bmg.E)

        return bmg
