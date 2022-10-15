from torch import Tensor
from torch.nn import functional as F

from chemprop.data.v2.dataloader import TrainingBatch
from chemprop.v2.models.modules import MessagePassingBlock
from chemprop.v2.models.models.base import MPNN


class MulticlassMPNN(MPNN):
    _DATASET_TYPE = "multiclass"
    _DEFAULT_CRITERION = "ce"
    _DEFAULT_METRIC = "ce"

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

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y.reshape((len(Y), self.n_tasks, self.n_classes))  # b x t x c

        return Y

    def predict_step(self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        Y = super().predict_step(batch, batch_idx, dataloader_idx)

        return Y.softmax(2)


class DirichletMulticlassMPNN(MulticlassMPNN):
    _DEFAULT_CRITERION = "dirichlet"

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
