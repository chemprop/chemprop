from abc import ABC, abstractmethod
import logging
from typing import Self

import numpy as np
from scipy.optimize import fmin
from scipy.special import expit, logit
import torch
from torch import Tensor

from chemprop.utils.registry import ClassRegistry

logger = logging.getLogger(__name__)


class CalibratorBase(ABC):
    """
    A base class for calibrating the predicted uncertainties.
    """

    @abstractmethod
    def fit(self, *args, **kwargs) -> Self:
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply(self, uncs: Tensor) -> Tensor:
        """
        Apply this calibrator to the input uncertainties.

        Parameters
        ----------
        uncs: Tensor
            a tensor containinig uncalibrated uncertainties

        Returns
        -------
        Tensor
            the calibrated uncertainties
        """


UncertaintyCalibratorRegistry = ClassRegistry[CalibratorBase]()


class RegressionCalibrator(CalibratorBase):
    """
    A class for calibrating the predicted uncertainties in regressions tasks.
    """

    @abstractmethod
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        """
        Fit calibration method for the calibration data.

        Parameters
        ----------
        preds: Tensor
            the predictions for regression tasks. It is a tensor of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        uncs: Tensor
            the predicted uncertainties (variance) of the shape of ``n x t``
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        self : RegressionCalibrator
            the fitted calibrator
        """


@UncertaintyCalibratorRegistry.register("zscaling")
class ZScalingCalibrator(RegressionCalibrator):
    """Calibrate regression datasets by applying a scaling value to the uncalibrated standard deviation,
    fitted by minimizing the negative-log-likelihood of a normal distribution around each prediction. [levi2022]_

    References
    ----------
    .. [levi2022] Levi, D.; Gispan, L.; Giladi, N.; Fetaya, E. "Evaluating and calibrating
    uncertainty prediction in regression tasks". Sensors, 2022, 22 (15), 5540.
    """

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        scalings = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j].numpy()
            uncs_j = uncs[:, j][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()
            errors = preds_j - targets_j

            def objective(scaler_value: float):
                scaled_vars = uncs_j * scaler_value**2
                nll = np.log(2 * np.pi * scaled_vars) / 2 + errors**2 / (2 * scaled_vars)
                return nll.sum()

            zscore = errors / np.sqrt(uncs_j)
            initial_guess = np.std(zscore)
            scalings.append(fmin(objective, x0=initial_guess, disp=False))

        self.scalings = torch.tensor(scalings)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return uncs * self.scalings**2


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(RegressionCalibrator):
    """Calibrate regression datasets that does not depend on a particular probability function form.

    This method is designed to be used with interval output. Uses the "CRUDE" method as described in [zelikman2020]_.

    Parameters
    ----------
    p: float
        The target qunatile, :math:`p \in [0, 1]`

    References
    ----------
    .. [zelikman2020] Zelikman, E.; Healy, C.; Zhou, S.; Avati, A. "CRUDE: calibrating regression uncertainty
    distributions empirically." arXiv preprint arXiv:2005.12496. https://doi.org/10.48550/arXiv.2005.12496
    """

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        if not 0 <= self.p <= 1:
            raise ValueError(f"arg `p` must be between 0 and 1. got: {p}.")

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        scalings = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            z = (preds_j - targets_j).abs() / (uncs_j).sqrt()
            scaling = torch.quantile(z, self.p, interpolation="lower")
            scalings.append(scaling)

        self.scalings = torch.tensor(scalings)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return uncs * self.scalings**2


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class ConformalRegressionCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("conformal-quantile-regression")
class ConformalQuantileRegressionCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


class BinaryClassificationCalibrator(CalibratorBase):
    """
    A class for calibrating the predicted uncertainties in binary classification tasks.
    """

    @abstractmethod
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        """
        Fit calibration method for the calibration data.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probability of class 1) of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        self : BinaryClassificationCalibrator
            the fitted calibrator
        """


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(BinaryClassificationCalibrator):
    """Calibrate classification datasets using the Platt scaling algorithm [guo2017]_, [platt1999]_.

    In [platt1999]_, Platt suggests using the number of positive and negative training examples to
    adjust the value of target probabilities used to fit the parameters.

    References
    ----------
    .. [guo2017] Guo, C.; Pleiss, G.; Sun, Y.; Weinberger, K. Q. "On calibration of modern neural
    networks". ICML, 2017. https://arxiv.org/abs/1706.04599
    .. [platt1999] Platt, J. "Probabilistic Outputs for Support Vector Machines and Comparisons to
    Regularized Likelihood Methods." Adv. Large Margin Classif. 1999, 10 (3), 61–74.
    """

    def fit(
        self, uncs: Tensor, targets: Tensor, mask: Tensor, training_targets: Tensor | None = None
    ) -> Self:
        if torch.any((targets[mask] != 0) & (targets[mask] != 1)):
            raise ValueError(
                "Platt scaling is only implemented for binary classification tasks! Input tensor "
                "must contain only 0's and 1's."
            )

        if training_targets is not None:
            logger.info(
                "Training targets were provided. Platt scaling for calibration uses a Bayesian "
                "correction to avoid training set overfitting. Now replacing calibration targets "
                "[0, 1] with adjusted values."
            )

            n_negative_examples = (training_targets == 0).sum(dim=0)
            n_positive_examples = (training_targets == 1).sum(dim=0)

            negative_target_bayes_MAP = (1 / (n_negative_examples + 2)).expand_as(targets)
            positive_target_bayes_MAP = (
                (n_positive_examples + 1) / (n_positive_examples + 2)
            ).expand_as(targets)

            targets = targets.float()
            targets[targets == 0] = negative_target_bayes_MAP[targets == 0]
            targets[targets == 1] = positive_target_bayes_MAP[targets == 1]
        else:
            logger.info("No training targets were provided. No Bayesian correction is applied.")

        xs = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            uncs_j = uncs[:, j][mask_j].numpy()
            targets_j = targets[:, j][mask_j].numpy()

            def objective(parameters):
                a, b = parameters
                scaled_uncs = expit(a * logit(uncs_j) + b)
                nll = -1 * np.sum(
                    targets_j * np.log(scaled_uncs) + (1 - targets_j) * np.log(1 - scaled_uncs)
                )
                return nll

            xs.append(fmin(objective, x0=[1, 0], disp=False))

        xs = np.vstack(xs)
        self.a, self.b = torch.tensor(xs).T.unbind(dim=0)

        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return torch.sigmoid(self.a * torch.logit(uncs) + self.b)


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(BinaryClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class ConformalMultilabelCalibrator(BinaryClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


class MulticlassClassificationCalibrator(CalibratorBase):
    """
    A class for calibrating the predicted uncertainties in multiclass classification tasks.
    """

    @abstractmethod
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        """
        Fit calibration method for the calibration data.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probabilities for each class) of the shape of ``n x t x c``, where ``n`` is the number of input
            molecules/reactions, ``t`` is the number of tasks, and ``c`` is the number of classes.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the fitting

        Returns
        -------
        self : MulticlassClassificationCalibrator
            the fitted calibrator
        """


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class ConformalMulticlassCalibrator(MulticlassClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class ConformalAdaptiveMulticlassCalibrator(MulticlassClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(MulticlassClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return
