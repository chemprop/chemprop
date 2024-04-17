from abc import abstractmethod

import torch
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from chemprop.data.collate import BatchMolGraph


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))

    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler):
        return cls(scaler.mean_, scaler.scale_)

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        pass


class ScaleTransform(TensorTransformBase):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return (X - self.mean) / self.scale


class UnscaleTransform(TensorTransformBase):
    def forward(self, X: Tensor) -> Tensor:
        if self.training:
            return X

        return X * self.scale + self.mean


class GraphTransform(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike, key: str, inds: ArrayLike):
        super().__init__()

        assert key in {"V", "E"}

        self.transform = TensorTransform(mean, scale)
        self.key = key
        self.inds = inds

    def from_standard_scaler(scaler: StandardScaler, key: str, inds: ArrayLike):
        return GraphTransform(scaler.mean_, scaler.scale_, key, inds)

    def forward(self, bmg: BatchMolGraph) -> BatchMolGraph:
        if self.training:
            return bmg

        X = getattr(bmg, self.key)
        X[:, self.inds] = self.transform(X[:, -self.inds])

        return bmg
