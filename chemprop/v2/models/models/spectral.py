from torch import Tensor, nn

from chemprop.v2.models.models.base import MPNN


class Exp(nn.Module):
    def forward(self, x: Tensor):
        return x.exp()


class SpectralMPNN(MPNN):
    _DATASET_TYPE = "spectral"
    _DEFAULT_CRITERION = "sid"
    _DEFAULT_METRIC = "sid"

    def __init__(self, *args, spectral_activation: str = "softplus", **kwargs):
        super().__init__(*args, **kwargs)
        act = nn.Softplus() if spectral_activation == "softplus" else Exp()
        self.ffn.add_module(f"{len(self.ffn)}", act)
