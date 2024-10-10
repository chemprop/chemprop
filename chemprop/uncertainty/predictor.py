from abc import ABC, abstractmethod
from typing import Iterable

from lightning import pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.models.model import MPNN
from chemprop.utils.registry import ClassRegistry


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
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


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
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        return


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


@UncertaintyPredictorRegistry.register("quantile-regression")
class QuantileRegressionPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        individual_preds = []
        for model in models:
            predss = trainer.predict(model, dataloader)
            individual_preds.append(torch.concat(predss, 0))
        stacked_preds = torch.stack(individual_preds).float()
        mean = stacked_preds[..., 0]
        interval = torch.mean(stacked_preds[..., 1], dim=0)
        return mean, interval
