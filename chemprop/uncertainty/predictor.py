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


@UncertaintyPredictorRegistry.register("none")
class NoUncertaintyPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        predss = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            predss.append(preds)
        return torch.stack(predss), None


@UncertaintyPredictorRegistry.register("mve")
class MVEPredictor(UncertaintyPredictor):
    """
    Class that estimates prediction means and variances (MVE). [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        mves = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            mves.append(preds)
        mves = torch.stack(mves, dim=0)
        mean, var = mves.unbind(dim=-1)
        return mean, var


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
        predss = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            predss.append(preds)
        return torch.stack(predss), torch.stack(predss)


@UncertaintyPredictorRegistry.register("evidential-total")
class EvidentialTotalPredictor(UncertaintyPredictor):
    """
    Class that predicts the total evidential uncertainty based on hyperparameters of
    the evidential distribution [amini2020]_.

    References
    -----------
    .. [amini2020] Amini, A.; Schwarting, W.; Soleimany, A.; Rus, D. "Deep Evidential Regression".
    NeurIPS, 2020. https://proceedings.neurips.cc/paper_files/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf

    """

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        uncs = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            uncs.append(preds)
        uncs = torch.stack(uncs)
        mean, v, alpha, beta = uncs.unbind(-1)
        total_uncs = (1 + 1 / v) * (beta / (alpha - 1))
        return mean, total_uncs


@UncertaintyPredictorRegistry.register("evidential-epistemic")
class EvidentialEpistemicPredictor(UncertaintyPredictor):
    """
    Class that predicts the epistemic evidential uncertainty based on hyperparameters of
    the evidential distribution.
    """

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        uncs = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            uncs.append(preds)
        uncs = torch.stack(uncs)
        mean, v, alpha, beta = uncs.unbind(-1)
        epistemic_uncs = (1 / v) * (beta / (alpha - 1))
        return mean, epistemic_uncs


@UncertaintyPredictorRegistry.register("evidential-aleatoric")
class EvidentialAleatoricPredictor(UncertaintyPredictor):
    """
    Class that predicts the aleatoric evidential uncertainty based on hyperparameters of
    the evidential distribution.
    """

    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        uncs = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            uncs.append(preds)
        uncs = torch.stack(uncs)
        mean, _, alpha, beta = uncs.unbind(-1)
        aleatoric_uncs = beta / (alpha - 1)
        return mean, aleatoric_uncs


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
        return torch.stack(meanss), torch.stack(varss)

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


@UncertaintyPredictorRegistry.register("classification-dirichlet")
class ClassificationDirichletPredictor(UncertaintyPredictor):
    def __call__(
        self, dataloader: DataLoader, models: Iterable[MPNN], trainer: pl.Trainer
    ) -> tuple[Tensor, Tensor]:
        uncs = []
        for model in models:
            preds = torch.concat(trainer.predict(model, dataloader), 0)
            uncs.append(preds)
        uncs = torch.stack(uncs, dim=0)
        y, u = uncs.unbind(dim=-1)
        return y, u


@UncertaintyPredictorRegistry.register("multiclass-dirichlet")
class MulticlassDirichletPredictor(UncertaintyPredictor):
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
