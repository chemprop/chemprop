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

            * multiclass classification: ``m x n x t x c``, where ``m`` is the number of models,
            ``n`` is the number of inputs, ``t`` is the number of tasks, and ``c`` is the number of classes.
        uncs : Tensor
            the predicted uncertainties, with shapes of ``m' x n x t`` for regression,
            or binary classification, and ``m' n x t x c`` for multiclass classification.

        .. note::
            The ``m`` and ``m'`` are different by definition. The ``m`` is the number of models,
            while the ``m'`` is the number of uncertainty estimations. For example, if two MVE
            or evidential models are provided, both ``m`` and ``m'`` are two. However, for an
            ensemble of two models, ``m'`` would be one (even though ``m = 2``).
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
    Class that predicts the uncertainty of predictions based on the variance in predictions among
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
            ensemble_preds.append(preds)
        stacked_preds = torch.stack(ensemble_preds).float()
        vars = torch.var(stacked_preds, dim=0, correction=0).unsqueeze(0)
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
    A :class:`DropoutPredictor` creates a virtual ensemble of models via Monte Carlo dropout with
    the provided model [1]_.

    References
    -----------
    .. [1] arXiv:1506.02142 [stat.ML]

    Parameters
    ----------
    ensemble_size: int
        The number of samples to draw for the ensemble.
    dropout: float | None
        The probability of dropping out units in the dropout layers. If unspecified,
        the training probability is used, which is prefered but not possible if the model was not
        trained with dropout (i.e. p=0).
    """

    def __init__(self, ensemble_size: int, dropout: None | float = None):
        self.ensemble_size = ensemble_size
        self.dropout = dropout

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        meanss, varss = [], []
        for model in models:
            self._setup_model(model)
            individual_preds = []

            for _ in range(self.ensemble_size):
                predss = trainer.predict(model, dataloader)
                preds = torch.concat(predss, 0)
                individual_preds.append(preds)

            stacked_preds = torch.stack(individual_preds, dim=0).float()
            means = torch.mean(stacked_preds, dim=0).unsqueeze(0)
            vars = torch.var(stacked_preds, dim=0, correction=0)
            self._restore_model(model)
            meanss.append(means)
            varss.append(vars)
        return torch.stack(means), torch.stack(vars)

    def _setup_model(self, model):
        model._predict_step = model.predict_step
        model.predict_step = self._predict_step(model)
        model.apply(self._change_dropout)

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
            module.train()

    def _change_dropout(self, module):
        if isinstance(module, torch.nn.Dropout):
            module._p = module.p
            if self.dropout:
                module.p = self.dropout

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
        mean, interval = stacked_preds.unbind(2)
        return mean, interval
