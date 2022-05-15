import torch
from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MoleculeModel


class RegressionMoleculeModel(MoleculeModel):
    """The RegressionMoleculeModel is just an alias for a base MoleculeModel"""


class MveRegressionMoleculeModel(RegressionMoleculeModel):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
    ):
        super().__init__(encoder, 2 * num_tasks, ffn_hidden_dim, ffn_num_layers)
        self.softplus = nn.Softplus()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)

        Y_mean, Y_var = torch.split(Y, Y.shape[1] // 2, 1)
        Y_var = self.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)


class EvidentialMoleculeModel(RegressionMoleculeModel):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
    ):
        super().__init__(encoder, 4 * num_tasks, ffn_hidden_dim, ffn_num_layers)

        self.softplus = nn.Softplus()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)

        means, lambdas, alphas, betas = torch.split(Y, Y.shape[1] // 4, dim=1)
        lambdas = self.softplus(lambdas)
        alphas = self.softplus(alphas) + 1
        betas = self.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)
