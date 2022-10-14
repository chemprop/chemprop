from torch import Tensor, nn
from torch.nn import functional as F

from chemprop.models.v2.modules import MessagePassingBlock
from chemprop.models.v2.models.base import MPNN


class MulticlassMPNN(MPNN):
    def __init__(
        self,
        encoder: MessagePassingBlock,
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
        encoder: MessagePassingBlock,
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

    @property
    def n_targets(self) -> int:
        return 2
        
    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = F.softplus(Y) + 1

        return Y
