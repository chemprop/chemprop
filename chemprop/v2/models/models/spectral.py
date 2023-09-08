from typing import Optional

from torch import Tensor, nn

from chemprop.v2.models.models.base import MPNN


class Exp(nn.Module):
    def forward(self, x: Tensor):
        return x.exp()


class SpectralMPNN(MPNN):
    _DATASET_TYPE = "spectral"
    _DEFAULT_CRITERION = "sid"
    _DEFAULT_METRIC = "sid"

    def __init__(self, *args, spectral_activation: Optional[str] = "softplus", **kwargs):
        super().__init__(*args, **kwargs)
        if spectral_activation == "exp":
            act = Exp()
        elif spectral_activation == "softplus" or spectral_activation is None:
            act = nn.Softplus()
        else:
            raise ValueError(
                f"Invalid spetral activation! got: {spectral_activation}, "
                "expected one of: {'softplus', 'exp', `None`}"
            )
        self.ffn.add_module(f"{len(self.ffn)}", act)
