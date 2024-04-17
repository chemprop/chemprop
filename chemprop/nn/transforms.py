from abc import abstractmethod

import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from chemprop.data.collate import BatchMolGraph


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike | None, scale: ArrayLike | None, pad: int = 0):
        super().__init__()

        if pad == 0:
            assert mean is not None
            assert scale is not None

        if pad > 0:

            if mean is None:
                mean = torch.zeros(pad)
            else:
                mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])

            if scale is None:
                scale = torch.ones(pad)
            else:
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

    @classmethod
    def from_standard_scaler(V_scaler: StandardScaler | None, E_scaler: StandardScaler | None, atom_fdim: int, bond_fdim: int) -> GraphTransform:
        V_mean = V_scaler.mean_ if V_scaler is not None else None
        V_scale = V_scaler.scale_ if V_scaler is not None else None
        E_mean = E_scaler.mean_ if E_scaler is not None else None
        E_scale = E_scaler.scale_ if E_scaler is not None else None
        return GraphTransform(V_mean, V_scale, atom_fdim, E_mean, E_scale, bond_fdim)

    def forward(self, bmg: BatchMolGraph) -> BatchMolGraph:
        if self.training:
            return bmg

        bmg.V = self.V_transform(bmg.V)
        bmg.E = self.E_transform(bmg.E)

        return bmg
