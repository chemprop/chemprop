from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor, nn

from chemprop.data.collate import BatchMolGraph


class _ScaleTransformMixin(nn.Module):
    def __init__(self, mean: ArrayLike, scale: ArrayLike, pad: int = 0):
        super().__init__()

        mean = torch.cat([torch.zeros(pad), torch.tensor(mean, dtype=torch.float)])
        scale = torch.cat([torch.ones(pad), torch.tensor(scale, dtype=torch.float)])

        if mean.shape != scale.shape:
            raise ValueError(
                f"uneven shapes for 'mean' and 'scale'! got: mean={mean.shape}, scale={scale.shape}"
            )

        self.register_buffer("mean", mean.unsqueeze(0))
        self.register_buffer("scale", scale.unsqueeze(0))

    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler, pad: int = 0):
        return cls(scaler.mean_, scaler.scale_, pad=pad)


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
    def __init__(self, V_transform: ScaleTransform, E_transform: ScaleTransform):
        super().__init__()

        self.V_transform = V_transform
        self.E_transform = E_transform

    def forward(self, bmg: BatchMolGraph) -> BatchMolGraph:
        if self.training:
            return bmg

        bmg.V = self.V_transform(bmg.V)
        bmg.E = self.E_transform(bmg.E)

        return bmg
