from typing import Iterable, Optional, Union
import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.models.v2.models.base import MPNN
from chemprop.models.v2.models.loss import (
    EvidentialLoss, LossFunction, MSELoss, BoundedMSELoss, MVELoss, get_loss
)
from chemprop.models.v2.models.metrics import Metric, RMSEMetric


class RegressionMPNN(MPNN):
    @MPNN.criterion.setter
    def criterion(self, criterion: Optional[Union[str, LossFunction]]) -> LossFunction:
        if criterion is None:
            criterion = MSELoss()
        elif isinstance(criterion, str):
            criterion = get_loss("regression", criterion)

        if not isinstance(criterion, (MSELoss, BoundedMSELoss)):
            raise ValueError(
                "Invalid regression criterion! "
                f"got: '{criterion}'. expected one of: ('mse', 'bounded')"
            )

        self.__criterion = criterion
    
    @MPNN.metrics.setter
    def metrics(self, metrics: Optional[Iterable[Union[str, LossFunction]]]) -> Iterable[Metric]:
        if metrics is None:
            metrics = [RMSEMetric]
        else:
            metrics = []

        self.__metrics = metrics


class MveRegressionMPNN(RegressionMPNN):    
    @RegressionMPNN.criterion.setter
    def criterion(self, criterion: Optional[Union[str, LossFunction]]) -> LossFunction:
        if criterion is None:
            criterion = MVELoss()
        elif isinstance(criterion, str):
            criterion = get_loss("regression", criterion)

        if not isinstance(criterion, MVELoss):
            raise ValueError(
                f"Invalid regression criterion! got: '{criterion}'. expected one of: ('mve')"
            )

        self.__criterion = criterion

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, *args, **kwargs) -> Tensor:
        Y = super().forward(*args, **kwargs)

        Y_mean, Y_var = torch.split(Y, Y.shape[1] // 2, 1)
        Y_var = F.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)


class EvidentialMPNN(RegressionMPNN):
    @RegressionMPNN.criterion.setter
    def criterion(self, criterion: Optional[Union[str, LossFunction]]) -> LossFunction:
        if criterion is None:
            criterion = EvidentialLoss()
        elif isinstance(criterion, str):
            criterion = get_loss("regression", criterion)

        if not isinstance(criterion, EvidentialLoss):
            raise ValueError(
                f"Invalid regression criterion! got: '{criterion}'. expected one of: ('evidential')"
            )

        self.__criterion = criterion

    @property
    def n_targets(self) -> int:
        return 4

    def forward(self, *args, **kwargs) -> Tensor:
        Y = super().forward(*args, **kwargs)

        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, dim=1)
        lambdas = F.softplus(lambdas)
        alphas = F.softplus(alphas) + 1
        betas = F.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)
