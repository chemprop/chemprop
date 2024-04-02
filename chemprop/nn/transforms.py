from sklearn.preprocessing import StandardScaler
import torch
from torch import nn, Tensor
from numpy.typing import ArrayLike

class OutputTransform(nn.Module):
    mean: Tensor
    scale: Tensor

    def __init__(self, mean: ArrayLike, scale: ArrayLike):
        super().__init__()
        
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float))

    def forward(self, Y: Tensor) -> Tensor:
        return Y if self.training else Y * self.scale + self.mean
    
    @classmethod
    def from_standard_scaler(cls, scaler: StandardScaler):
        return cls(scaler.mean_, scaler.scale_)