from torch import Tensor, nn

from chemprop.v2.data.dataloader import TrainingBatch
from chemprop.v2.models.models.base import MPNN


class ClassificationMPNN(MPNN):
    _DATASET_TYPE = "classification"
    _DEFAULT_METRIC = "auroc"


class BinaryClassificationMPNN(ClassificationMPNN):
    _DEFAULT_CRITERION = "bce"

    def predict_step(self, batch: TrainingBatch, batch_idx: int, dataloader_idx: int = 0):
        Y = super().predict_step(batch, batch_idx, dataloader_idx)

        return Y.sigmoid()


class DirichletClassificationMPNN(ClassificationMPNN):
    _DEFAULT_CRITERION = "dirichlet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, *args) -> Tensor:
        Y = super().forward(*args)
        Y = self.softplus(Y) + 1

        return Y
