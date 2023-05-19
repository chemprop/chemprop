import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.v2.models.models.base import MPNN


class RegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "mse"
    _DEFAULT_METRIC = "rmse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #NOTE(degraff): these msut be tensors for multitask scaling
        self.scale = 1
        self.loc = 0

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, ...]:
        Y = super().predict_step(*args, **kwargs)[0]
        Y = Y * self.scale + self.loc

        return (Y,)


class MveRegressionMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "mve"

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f=X_f)

        Y_mean, Y_var = Y.split(Y.shape[1] // 2, 1)
        Y_var = F.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, ...]:
        Y = MPNN.predict_step(self, *args, **kwargs)[0]
        Y_mean, Y_var = Y.split(Y.shape[1] // 2, dim=1)
        Y_mean = Y_mean * self.scale + self.loc
        Y_var = Y_var * self.scale**2

        return Y_mean, Y_var


class EvidentialMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "evidential"

    @property
    def n_targets(self) -> int:
        return 4

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)

        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, dim=1)
        lambdas = F.softplus(lambdas)
        alphas = F.softplus(alphas) + 1
        betas = F.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        Y = MPNN.predict_step(self, *args, **kwargs)[0]
        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, 1)
        means = means * self.scale + self.loc
        lambdas = lambdas * self.scale**2

        return means, lambdas, alphas, betas
