import torch
from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MPNN


class RegressionMPNN(MPNN):
    """The RegressionMoleculeModel is just an alias for a base MPNN"""


class MveRegressionMPNN(RegressionMPNN):
    def __init__(
        self,
        encoder: MPNEncoder,
        n_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            encoder, n_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)

        Y_mean, Y_var = torch.split(Y, Y.shape[1] // 2, 1)
        Y_var = self.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)


class EvidentialMPNN(RegressionMPNN):
    def __init__(
        self,
        encoder: MPNEncoder,
        n_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            encoder, n_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 4

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)

        means, lambdas, alphas, betas = torch.split(Y, Y.shape[1] // 4, dim=1)
        lambdas = self.softplus(lambdas)
        alphas = self.softplus(alphas) + 1
        betas = self.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)
