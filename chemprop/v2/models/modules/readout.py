from typing import Protocol

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from chemprop.v2.models import loss
from chemprop.v2.models.modules.ffn import FFN, SimpleFFN
from chemprop.v2.utils import ClassRegistry

ReadoutRegistry = ClassRegistry()


class ReadoutProto(Protocol):
    n_tasks: int
    """the number of tasks `t` to predict for each input"""
    n_targets: int
    """the number of targets `s` to predict for each task `t`"""
    criterion: loss.LossFunction
    """the loss function to use for training"""
    
    def forward(self, Z: Tensor) -> Tensor:
        pass

    def train_step(self, Z: Tensor) -> Tensor:
        pass


class ReadoutFFN(FFN, ReadoutProto):
    pass


class ReadoutFFNBase(SimpleFFN, ReadoutFFN):
    _default_criterion: loss.LossFunction

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: loss.LossFunction | None = None,
    ):
        super().__init__(
            input_dim, n_tasks * self.n_targets, hidden_dim, n_layers, dropout, activation
        )
        self.n_tasks = n_tasks
        self.criterion = criterion

    @property
    def criterion(self) -> loss.LossFunction:
        return self.__criterion

    @criterion.setter
    def criterion(self, criterion: loss.LossFunction | None):
        self.__criterion = criterion or self._default_criterion


@ReadoutRegistry.register("regression")
class RegressionFFN(ReadoutFFNBase):
    n_targets = 1
    _default_criterion = loss.MSELoss()

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        loc: float | Tensor = 0,
        scale: float | Tensor = 1,
    ):
        super().__init__(input_dim, n_tasks, hidden_dim, n_layers, dropout, activation)

        self.loc = nn.Parameter(torch.tensor(loc).view(-1, 1), False)
        self.scale = nn.Parameter(torch.tensor(scale).view(-1, 1), False)

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return self.scale * Y + self.loc

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@ReadoutRegistry.register("regression-mve")
class MveFFN(RegressionFFN):
    n_targets = 2
    _default_criterion = loss.MVELoss()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        mean, var = torch.chunk(Y, self.n_targets, 1)

        mean = self.scale * mean + self.loc
        var = var * self.scale**2

        return torch.cat((mean, var), 1)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        mean, var = torch.chunk(Y, self.n_targets, 1)
        var = F.softplus(var)

        return torch.cat((mean, var), 1)


@ReadoutRegistry.register("regression-evidential")
class EvidentialFFN(RegressionFFN):
    n_targets = 4
    _default_criterion = loss.EvidentialLoss()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, 1)

        mean = self.scale * mean + self.loc
        v = v * self.scale**2

        return torch.cat((mean, v, alpha, beta), 1)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        mean, v, alpha, beta = torch.chunk(Y, self.n_targets, 1)

        v = F.softplus(v)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)

        return torch.cat((mean, v, alpha, beta), 1)


class BinaryClassificationFFNBase(ReadoutFFNBase):
    pass


@ReadoutRegistry.register("classification")
class BinaryClassificationFFN(BinaryClassificationFFNBase):
    n_targets = 1
    _default_criterion = loss.BCELoss()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return Y.sigmoid()

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@ReadoutRegistry.register("classification-dirichlet")
class BinaryDirichletFFN(BinaryClassificationFFNBase):
    n_targets = 2
    _default_criterion = loss.BinaryDirichletLoss()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        alpha, beta = torch.chunk(Y, 2, 1)

        return beta / (alpha + beta)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        F.softplus(Y) + 1


@ReadoutRegistry.register("multiclass")
class MulticlassClassificationFFN(ReadoutFFNBase):
    n_targets = 1
    _default_criterion = loss.CrossEntropyLoss()

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        n_classes: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: loss.LossFunction | None = None,
    ):
        super().__init__(input_dim, n_tasks, hidden_dim, n_layers, dropout, activation, criterion)

        self.n_classes = n_classes

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        Y = Y.reshape(Y.shape[0], -1, self.n_classes)

        return Y.softmax(-1)

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z).reshape(Z.shape[0], -1, self.n_classes)


@ReadoutRegistry.register("multiclass-dirichlet")
class MulticlassDirichletFFN(MulticlassClassificationFFN):
    _default_criterion = loss.MulticlassDirichletLoss()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(Y.shape[0], -1, self.n_classes)

        Y = Y.softmax(-1)
        Y = F.softplus(Y) + 1

        alpha = Y
        Y = Y / Y.sum(-1, keepdim=True)

        return torch.cat((Y, alpha), 1)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(Y.shape[0], -1, self.n_classes)

        return F.softplus(Y) + 1


class Exp(nn.Module):
    def forward(self, X: Tensor):
        return X.exp()


@ReadoutRegistry.register("spectral")
class SpectralFFN(ReadoutFFNBase):
    n_targets = 1
    _default_criterion = loss.SIDLoss()

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: loss.LossFunction | None = None,
        spectral_activation: str | None = "softplus",
    ):
        super().__init__(input_dim, n_tasks, hidden_dim, n_layers, dropout, activation, criterion)

        match spectral_activation:
            case "exp":
                spectral_activation = Exp()
            case "softplus" | None:
                spectral_activation = nn.Softplus()
            case _:
                raise ValueError(
                    f"Unknown spectral activation: {spectral_activation}. "
                    "Expected one of 'exp', 'softplus' or None."
                )

        self.ffn.add_module("spectral_activation", spectral_activation)
