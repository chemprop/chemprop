from torch import Tensor, nn

from chemprop.models.v2.encoders.base import MPNEncoder
from chemprop.models.v2.models.base import MoleculeModel


class MulticlassMoleculeModel(MoleculeModel):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        num_classes: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
    ):
        super().__init__(encoder, num_tasks * num_classes, ffn_hidden_dim, ffn_num_layers)

        self.num_classes = num_classes
        self.softmax = nn.Softmax(2)

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y.reshape((len(Y), self.num_tasks, self.num_classes))  # b x t x c
        # Z = self.softmax(Z)

        return Y


class DirichletMulticlassModel(MulticlassMoleculeModel):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        num_classes: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
    ):
        super().__init__(encoder, 2 * num_tasks, num_classes, ffn_hidden_dim, ffn_num_layers)

        self.softplus = nn.Softplus()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
