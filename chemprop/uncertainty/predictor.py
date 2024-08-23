from abc import ABC, abstractmethod
from typing import Iterable

from lightning import pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.models.model import MPNN
from chemprop.utils.registry import ClassRegistry


class UncertaintyPredictor(ABC):
    """
    A class for making model predictions and associated predictions of uncertainty.
    """

    def __call__(self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer):
        return self._calc_prediction_uncertainty(dataloader, models, trainer)

    @abstractmethod
    def _calc_prediction_uncertainty(
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

            * regression/binary classification: ``n x t x m``

            * multiclass classification: ``n x t x c x m``
            where ``n`` is the number of inputs, ``t`` is the number of tasks, ``c`` is the number of classes, and ``m`` is the number of models.
        uncs : Tensor
            the predicted uncertainties, with shapes of ``n x t`` for regression
            or binary classification, and ``n x t x c`` for multiclass classification.
        """


UncertaintyPredictorRegistry = ClassRegistry[UncertaintyPredictor]()


@UncertaintyPredictorRegistry.register(None)
class NoUncertaintyPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("mve")
class MVEPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("ensemble")
class EnsemblePredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("classification")
class ClassPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-total")
class EvidentialTotalPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-epistemic")
class EvidentialEpistemicPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("dropout")
class DropoutPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("spectra-roundrobin")
class RoundRobinSpectraPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("dirichlet")
class DirichletPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-regression")
class ConformalRegressionPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        ...
        return
