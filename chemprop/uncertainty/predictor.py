from abc import abstractmethod
from typing import Iterable

from lightning import pytorch as pl
from torch import Tensor
from torch.utils.data import DataLoader

from chemprop.models.model import MPNN
from chemprop.utils.registry import ClassRegistry


class UncertaintyPredictor:
    def __call__(self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer):
        return self._calc_prediction_uncertainty(dataloader, models, trainer)

    @abstractmethod
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        """
        Calculate the uncalibrated predictions and uncertainties for the dataloader.
        """
        pass


UncertaintyPredictorRegistry = ClassRegistry[UncertaintyPredictor]()


@UncertaintyPredictorRegistry.register(None)
class NoUncertaintyPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("mve")
class MVEPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("ensemble")
class EnsemblePredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("classification")
class ClassPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-total")
class EvidentialTotalPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-epistemic")
class EvidentialEpistemicPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("dropout")
class DropoutPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("spectra-roundrobin")
class RoundRobinSpectraPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("dirichlet")
class DirichletPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return


@UncertaintyPredictorRegistry.register("conformal-regression")
class ConformalRegressionPredictor(UncertaintyPredictor):
    def _calc_prediction_uncertainty(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> Tensor:
        ...
        return
