from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MPNN


class ClassificationMPNN(MPNN):
    """Single-task regresssion/classification networks are architecturally identical"""


class DirichletClassificationMPNN(ClassificationMPNN):
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
            encoder, 2 * n_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.n_targets = n_tasks
        self.softplus = nn.Softplus()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
