from torch import Tensor, nn

from chemprop.models.v2.models.base import MPNN


class Exp(nn.Module):
    def forward(self, x: Tensor):
        return x.exp()


class SpectralMPNN(MPNN):
    def __init__(self, *args, spectral_activation: str = "softplus", **kwargs):
        super().__init__(*args, **kwargs)
        act = nn.Softplus() if spectral_activation == "softplus" else Exp()
        self.ffn.add_module(f"{len(self.ffn)}", act)
