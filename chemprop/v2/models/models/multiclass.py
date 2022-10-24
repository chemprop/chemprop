from torch import Tensor, nn

from chemprop.v2.data.dataloader import TrainingBatch
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

    def forward(self, *args, **kwargs) -> Tensor:
        Y = super().forward(*args, **kwargs)
        Y.reshape(-1, self.n_tasks * self.n_targets, self.n_classes)

        return Y

    def predict_step(
        self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, ...]:
        Y = super().predict_step(batch, batch_idx, dataloader_idx)[0]

        return (Y.softmax(2),)


class DirichletMulticlassMPNN(MulticlassMPNN):
    _DEFAULT_CRITERION = "dirichlet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.softplus = nn.Softplus()

    def forward(self, *args, **kwargs) -> Tensor:
        Y = super().forward(*args, **kwargs)

        return self.softplus(Y) + 1

    def predict_step(
        self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> tuple[Tensor, ...]:
        alphas = MPNN.predict_step(self, batch, batch_idx, dataloader_idx)[0]
        preds = alphas / alphas.sum(2, keepdim=True)

        return preds, alphas
