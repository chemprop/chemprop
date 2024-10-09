from abc import ABC, abstractmethod
from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.models.model import MPNN
from chemprop.utils.registry import ClassRegistry
from chemprop.nn.predictors import MulticlassClassificationFFN


class UncertaintyPredictor(ABC):
    """A helper class for making model predictions and associated uncertainty predictions."""

    @abstractmethod
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate the uncalibrated predictions and uncertainties for the dataloader.

        dataloader: DataLoader
            the dataloader used for model predictions and uncertainty predictions
        models: Iterable[MPNN]
            the models used for model predictions and uncertainty predictions
        trainer: pl.Trainer
            an instance of the :class:`~lightning.pytorch.trainer.trainer.Trainer` used to manage model inference

        Returns
        -------
        preds : Tensor
            the model predictions, with shape varying by task type:

            * regression/binary classification: ``m x n x t``

            * multiclass classification: ``m x n x t x c``, where ``m`` is the number of models, ``n`` is the number of inputs, ``t`` is the number of tasks, and ``c`` is the number of classes.
        uncs : Tensor
            the predicted uncertainties, with shapes of ``n x t`` for regression
            or binary classification, and ``n x t x c`` for multiclass classification.
        """


UncertaintyPredictorRegistry = ClassRegistry[UncertaintyPredictor]()


@UncertaintyPredictorRegistry.register(None)
class NoUncertaintyPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("mve")
class MVEPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("ensemble")
class EnsemblePredictor(UncertaintyPredictor):
    """
    Class that predicts uncertainty for predictions based on the variance in predictions among
    an ensemble's submodels.
    """
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        if len(models) <= 1:
            raise ValueError(
                "Ensemble method for uncertainty is only available when multiple models are provided."
            )
        ensemble_preds = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            if isinstance(model.predictor, MulticlassClassificationFFN):
                preds = torch.argmax(preds, dim=-1)
            ensemble_preds.append(preds)
        stacked_preds = torch.stack(ensemble_preds).float()
        vars = torch.var(stacked_preds, dim=0, correction=0)
        return stacked_preds, vars


@UncertaintyPredictorRegistry.register("classification")
class ClassPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("evidential-total")
class EvidentialTotalPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("evidential-epistemic")
class EvidentialEpistemicPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("dropout")
class DropoutPredictor(UncertaintyPredictor):
    """
    A :class:`DropoutPredictor` creates a virtual ensemble of via Monte Carlo dropout with the provided models [1]_.

    References
    -----------
    .. [1] arXiv:1506.02142 [stat.ML]

    Parameters
    ----------
    ensemble_size: int
        The number of samples to draw for the ensemble.
    dropout: float | None
        The probability of dropping out units in the dropout layers. If unspecified,
        dropout will be performed using training probability.
    """

    def __init__(self, ensemble_size: int, dropout : None | float = None):
        self.ensemble_size = ensemble_size
        self.dropout = dropout

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        if len(models) != 1:
            raise ValueError(
                "Dropout method for uncertainty only takes exactly one model."
            )
        model = next(iter(models))
        self._setup_predict_wrapper(model)
        individual_preds = []

        for _ in range(self.ensemble_size):
            predss = trainer.predict(model, dataloader)
            preds = torch.concat(predss, 0)
            individual_preds.append(preds)

        stacked_preds = torch.stack(individual_preds, dim=0).float()
        means = torch.mean(stacked_preds, dim=0).unsqueeze(-1)
        vars = torch.var(stacked_preds, dim=0, correction=0)

        self._restore_model(model)
        return means, vars

    def _setup_predict_wrapper(self, model):
        model._predict_step = model.predict_step
        model.predict_step = self._predict_step(model)

    def _restore_model(self, model):
        model.predict_step = model._predict_step
        del model._predict_step
        model.apply(self._restore_dropout)

    def _predict_step(self, model):
        def _wrapped_predict_step(*args, **kwargs):
            model.apply(self._activate_dropout)
            return model._predict_step(*args, **kwargs)

        return _wrapped_predict_step

    def _activate_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            module._p = module.p
            if self.dropout:
                module.p = self.dropout
            module.train()

    def _restore_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            module.p = module._p
            del module._p


@UncertaintyPredictorRegistry.register("spectra-roundrobin")
class RoundRobinSpectraPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("dirichlet")
class DirichletPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


@UncertaintyPredictorRegistry.register("conformal-regression")
class ConformalRegressionPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return
