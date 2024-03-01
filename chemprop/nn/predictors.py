from abc import abstractmethod

from lightning.pytorch.core.mixins import HyperparametersMixin
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from chemprop.nn.loss import *
from chemprop.nn.metrics import *
from chemprop.nn.ffn import MLP

from chemprop.nn.hparams import HasHParams
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.utils import ClassRegistry

__all__ = [
    "Predictor",
    "PredictorRegistry",
    "RegressionFFN",
    "MveFFN",
    "EvidentialFFN",
    "BinaryClassificationFFNBase",
    "BinaryClassificationFFN",
    "BinaryDirichletFFN",
    "MulticlassClassificationFFN",
    "MulticlassDirichletFFN",
    "SpectralFFN",
]


class Predictor(nn.Module, HasHParams):
    r"""A :class:`Predictor` is a protocol that defines a differentiable function
    :math:`f` : \mathbb R^d \mapsto \mathbb R^o"""

    input_dim: int
    """the input dimension"""
    output_dim: int
    """the output dimension"""
    n_tasks: int
    """the number of tasks `t` to predict for each input"""
    n_targets: int
    """the number of targets `s` to predict for each task `t`"""
    criterion: LossFunction
    """the loss function to use for training"""

    @abstractmethod
    def forward(self, Z: Tensor) -> Tensor:
        pass

    @abstractmethod
    def train_step(self, Z: Tensor) -> Tensor:
        pass


PredictorRegistry = ClassRegistry[Predictor]()


class _FFNPredictorBase(Predictor, HyperparametersMixin):
    """A :class:`_FFNPredictorBase` is the base class for all :class:`Predictor`\s that use an
    underlying :class:`SimpleFFN` to map the learned fingerprint to the desired output."""

    _default_criterion: LossFunction
    _default_metric: Metric

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hparams["cls"] = self.__class__

        self.ffn = MLP(
            input_dim, n_tasks * self.n_targets, hidden_dim, n_layers, dropout, activation
        )
        self.criterion = criterion or self._default_criterion

    @property
    def input_dim(self) -> int:
        return self.ffn.input_dim

    @property
    def output_dim(self) -> int:
        return self.ffn.output_dim

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)

    def train_step(self, Z: Tensor) -> Tensor:
        return self.ffn(Z)


@PredictorRegistry.register("regression")
class RegressionFFN(_FFNPredictorBase):
    n_targets = 1
    _default_criterion = MSELoss()
    _default_metric = MSEMetric()

    def __init__(
        self,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
        loc: float | Tensor = 0.0,
        scale: float | Tensor = 1.0,
    ):
        super().__init__(n_tasks, input_dim, hidden_dim, n_layers, dropout, activation, criterion)

        if isinstance(loc, float):
            loc = torch.ones(1, self.n_tasks) * loc
        else:
            loc = torch.tensor(loc).view(1, -1)
        self.register_buffer("loc", loc)

        if isinstance(scale, float):
            scale = torch.ones(1, self.n_tasks) * scale
        else:
            scale = torch.tensor(scale).view(1, -1)
        self.register_buffer("scale", scale)

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return self.scale * Y + self.loc

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@PredictorRegistry.register("regression-mve")
class MveFFN(RegressionFFN):
    n_targets = 2
    _default_criterion = MVELoss()

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


@PredictorRegistry.register("regression-evidential")
class EvidentialFFN(RegressionFFN):
    n_targets = 4
    _default_criterion = EvidentialLoss()

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


class BinaryClassificationFFNBase(_FFNPredictorBase):
    pass


@PredictorRegistry.register("classification")
class BinaryClassificationFFN(BinaryClassificationFFNBase):
    n_targets = 1
    _default_criterion = BCELoss()
    _default_metric = AUROCMetric(task="binary")

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        return Y.sigmoid()

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z)


@PredictorRegistry.register("classification-dirichlet")
class BinaryDirichletFFN(BinaryClassificationFFNBase):
    n_targets = 2
    _default_criterion = BinaryDirichletLoss()
    _default_metric = AUROCMetric(task="binary")

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        alpha, beta = torch.chunk(Y, 2, 1)

        return beta / (alpha + beta)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)

        F.softplus(Y) + 1


@PredictorRegistry.register("multiclass")
class MulticlassClassificationFFN(_FFNPredictorBase):
    n_targets = 1
    _default_criterion = CrossEntropyLoss()
    _default_metric = CrossEntropyMetric()

    def __init__(
        self,
        n_classes: int,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0,
        activation: str = "relu",
        criterion: LossFunction | None = None,
    ):
        super().__init__(
            n_tasks * n_classes, input_dim, hidden_dim, n_layers, dropout, activation, criterion
        )

        self.n_classes = n_classes

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        Y = Y.reshape(Y.shape[0], -1, self.n_classes)

        return Y.softmax(-1)

    def train_step(self, Z: Tensor) -> Tensor:
        return super().forward(Z).reshape(Z.shape[0], -1, self.n_classes)


@PredictorRegistry.register("multiclass-dirichlet")
class MulticlassDirichletFFN(MulticlassClassificationFFN):
    _default_criterion = MulticlassDirichletLoss()
    _default_metric = CrossEntropyMetric()

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, self.n_classes)

        Y = Y.softmax(-1)
        Y = F.softplus(Y) + 1

        alpha = Y
        Y = Y / Y.sum(-1, keepdim=True)

        return torch.cat((Y, alpha), 1)

    def train_step(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z).reshape(len(Z), -1, self.n_classes)

        return F.softplus(Y) + 1


class _Exp(nn.Module):
    def forward(self, X: Tensor):
        return X.exp()


@PredictorRegistry.register("spectral")
class SpectralFFN(_FFNPredictorBase):
    n_targets = 1
    _default_criterion = SIDLoss()
    _default_metric = SIDMetric()

    def __init__(self, *args, spectral_activation: str | None = "softplus", **kwargs):
        super().__init__(*args, **kwargs)

        match spectral_activation:
            case "exp":
                spectral_activation = _Exp()
            case "softplus" | None:
                spectral_activation = nn.Softplus()
            case _:
                raise ValueError(
                    f"Unknown spectral activation: {spectral_activation}. "
                    "Expected one of 'exp', 'softplus' or None."
                )

        self.ffn.add_module("spectral_activation", spectral_activation)

    def forward(self, Z: Tensor) -> Tensor:
        Y = super().forward(Z)
        Y = self.ffn.spectral_activation(Y)
        return Y / Y.sum(1, keepdim=True)

    train_step = forward
