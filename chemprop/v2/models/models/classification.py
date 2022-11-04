from torch import Tensor, nn

from chemprop.v2.models.models.base import MPNN


class ClassificationMPNN(MPNN):
    _DATASET_TYPE = "classification"
    _DEFAULT_METRIC = "auroc"


class BinaryClassificationMPNN(ClassificationMPNN):
    _DEFAULT_CRITERION = "bce"

    def predict_step(self, *args, **kwargs) -> tuple[Tensor]:
        Y = super().predict_step(*args, **kwargs)[0]

        return (Y.sigmoid(),)


class DirichletClassificationMPNN(ClassificationMPNN):
    _DEFAULT_CRITERION = "dirichlet"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.softplus = nn.Softplus()

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, *args, **kwargs) -> Tensor:
        Y = super().forward(*args, **kwargs)

        return self.softplus(Y) + 1

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        alphas = super().predict_step(*args, **kwargs)[0]

        alphas = alphas.reshape(-1, self.n_tasks, 2)
        preds = alphas[..., 1] / alphas.sum(-1)

        return preds, alphas
