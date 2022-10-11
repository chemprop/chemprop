from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MPNN


class MulticlassMPNN(MPNN):
    def __init__(
        self,
        encoder: MPNEncoder,
        n_tasks: int,
        n_classes: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            encoder, n_tasks * n_classes, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.softmax = nn.Softmax(2)

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y.reshape((len(Y), self.n_tasks, self.n_classes))  # b x t x c
        # Z = self.softmax(Z)

        return Y


class DirichletMulticlassMPNN(MulticlassMPNN):
    def __init__(
        self,
        encoder: MPNEncoder,
        n_tasks: int,
        n_classes: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            encoder, n_tasks, n_classes, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )
        self.n_tasks = n_tasks
        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 2
        
    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
