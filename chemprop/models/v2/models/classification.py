from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MPNN


class ClassificationMPNN(MPNN):
    """A `ClassificationMPNN` is an alias for a base `MPNN`"""


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
            encoder, n_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
