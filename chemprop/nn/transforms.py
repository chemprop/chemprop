from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, Tensor
from numpy.typing import ArrayLike
from lightning.pytorch.core.mixins import HyperparametersMixin

from chemprop.nn.hparams import HasHParams

class OutputTransform(nn.Module):
    mean: Tensor
    scale: Tensor

    def __init__(self, mean: ArrayLike, scale: ArrayLike):
        super().__init__()
        
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, Y: Tensor) -> Tensor:
        return Y if self.training else Y * self.scale + self.mean
    
    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler):
        return cls(scaler.mean_, scaler.scale_)