from torch import Tensor, nn
from chemprop.models.v2.encoders.base import MPNEncoder

from chemprop.models.v2.models.base import MoleculeModel


class ClassificationMoleculeModel(MoleculeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sigmoid = nn.Sigmoid()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        # Y = self.sigmoid(Y)
        return Y


class DirichletClassificationModel(ClassificationMoleculeModel):
    def __init__(
        self,
        encoder: MPNEncoder,
        num_tasks: int,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__(
            encoder, 2 * num_tasks, ffn_hidden_dim, ffn_num_layers, dropout, activation
        )

        self.softplus = nn.Softplus()

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
