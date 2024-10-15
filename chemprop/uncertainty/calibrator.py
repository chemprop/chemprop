from abc import ABC, abstractmethod
import logging
import math
from typing import Self
import warnings

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
            the predicted uncertainties of the shape of ``n x t``
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
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("tscaling")
class TScalingCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(RegressionCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return


@UncertaintyCalibratorRegistry.register("conformal-regression")
class RegressionConformalCalibrator(RegressionCalibrator):
    r"""Conformalize quantiles to make the interval :math:`[\hat{t}_{\alpha/2}(x),\hat{t}_{1-\alpha/2}(x)]` to have
    approximately :math:`1-\alpha` coverage. [1]_

    .. math::
        s(x, y) &= \max \left\{ \hat{t}_{\alpha/2}(x) - y, y - \hat{t}_{1-\alpha/2}(x) \right\}

        \hat{q} &= Q(s_1, \ldots, s_n; \left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil)

        C(x) &= \left[ \hat{t}_{\alpha/2}(x) - \hat{q}, \hat{t}_{1-\alpha/2}(x) + \hat{q} \right]

    where :math:`s` is the nonconformity score as the difference between :math:`y` and its nearest quantile.
    :math:`\hat{t}_{\alpha/2}(x)` and :math:`\hat{t}_{1-\alpha/2}(x)` are the predicted quantiles from a quantile
    regression model.

    .. note::
        The algorithm is specifically designed for quantile regression model. Intuitively, the set :math:`C(x)` just
        grows or shrinks the distance between the quantiles by :math:`\hat{q}` to achieve coverage. However, this
        function can also be applied to regression model without quantiles being provided. In this case, both
        :math:`\hat{t}_{\alpha/2}(x)` and :math:`\hat{t}_{1-\alpha/2}(x)` are the same as :math:`\hat{y}`. Then, the
        interval would be the same for every data point (i.e., :math:`\left[-\hat{q}, \hat{q} \right]`).

    Parameters
    ----------
    alpha: float
        The error rate, :math:`\alpha \in [0, 1]`

    Returns
    -------
    Tensor
        the calibrated intervals

    References
    ----------
    .. [1] Angelopoulos, A.N.; Bates, S.; "A Gentle Introduction to Conformal Prediction and Distribution-Free
        Uncertainty Quantification." arXiv Preprint 2021, https://arxiv.org/abs/2107.07511
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        self.bounds = torch.tensor([-1 / 2, 1 / 2]).view(-1, 1)
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"arg `alpha` must be between 0 and 1. got: {alpha}.")

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        self.qhats = []
        for j in range(preds.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            preds_j = preds[:, j][mask_j]
            interval_j = uncs[:, j][mask_j]

            interval_bounds = self.bounds * interval_j.unsqueeze(0)
            pred_bounds = preds_j.unsqueeze(0) + interval_bounds

            calibration_scores = torch.max(pred_bounds[0] - targets_j, targets_j - pred_bounds[1])

            num_data = targets_j.shape[0]
            if self.alpha >= 1 / (num_data + 1):
                q_level = math.ceil((num_data + 1) * (1 - self.alpha)) / num_data
            else:
                q_level = 1
                warnings.warn(
                    "The error rate (i.e., alpha) is smaller than 1 / (number of data + 1), so the 1 - alpha quantile is set to 1, but this only ensures that the coverage is trivially satisfied."
                )
            qhat = torch.quantile(calibration_scores, q_level, interpolation="higher")
            self.qhats.append(qhat)

        self.qhats = torch.tensor(self.qhats)
        return self

    def apply(self, uncs: Tensor) -> tuple[Tensor, Tensor]:
        cal_intervals = uncs + 2 * self.qhats

        return cal_intervals


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

            # if is_atom_bond_targets: # Not yet implemented

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
class MultilabelConformalCalibrator(BinaryClassificationCalibrator):
    r"""Creates conformal in-set and conformal out-set such that, for :math:`1-\alpha` proportion of datapoints,
    the set of labels is bounded by the in- and out-sets [1]_:

    .. math::
        \Pr \left(
            \hat{\mathcal C}_{\text{in}}(X) \subseteq \mathcal Y \subseteq \hat{\mathcal C}_{\text{out}}(X)
        \right) \geq 1 - \alpha,

    where the in-set :math:`\hat{\mathcal C}_\text{in}` is contained by the set of true labels :math:`\mathcal Y` and
    :math:`\mathcal Y` is contained within the out-set :math:`\hat{\mathcal C}_\text{out}`.

    Parameters
    ----------
    alpha: float
        The error rate, :math:`\alpha \in [0, 1]`

    References
    ----------
    .. [1] Cauchois, M.; Gupta, S.; Duchi, J.; "Knowing What You Know: Valid and Validated Confidence Sets
        in Multiclass and Multilabel Prediction." arXiv Preprint 2020, https://arxiv.org/abs/2004.10181
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"arg `alpha` must be between 0 and 1. got: {alpha}.")

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        r"""
        Compute nonconformity score as the negative of the predicted probability.

        .. math::
            s_i = -\hat{f}(X_i)_{Y_i}
        """
        return -preds

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        if targets.shape[1] < 2:
            raise ValueError(
                f"the number of tasks should be larger than 1! got: {targets.shape[1]}."
            )

        has_zeros = torch.any(targets == 0, dim=1)
        index_zeros = targets[has_zeros] == 0
        scores_in = self.nonconformity_scores(uncs[has_zeros])
        masked_scores_in = scores_in * index_zeros.float() + torch.where(
            index_zeros, torch.zeros_like(scores_in), torch.tensor(float("inf"))
        )
        calibration_scores_in = torch.min(
            masked_scores_in.masked_fill(~mask, float("inf")), dim=1
        ).values

        has_ones = torch.any(targets == 1, dim=1)
        index_ones = targets[has_ones] == 1
        scores_out = self.nonconformity_scores(uncs[has_ones])
        masked_scores_out = scores_out * index_ones.float() + torch.where(
            index_ones, torch.zeros_like(scores_out), torch.tensor(float("-inf"))
        )
        calibration_scores_out = torch.max(
            masked_scores_out.masked_fill(~mask, float("-inf")), dim=1
        ).values

        self.tout = torch.quantile(
            calibration_scores_out, 1 - self.alpha / 2, interpolation="higher"
        )
        self.tin = torch.quantile(calibration_scores_in, self.alpha / 2, interpolation="higher")
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        scores = self.nonconformity_scores(uncs)

        cal_preds_in = (scores <= self.tin).int()
        cal_preds_out = (scores <= self.tout).int()
        cal_preds_in_out = torch.cat((cal_preds_in, cal_preds_out), dim=1)

        return cal_preds_in_out


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
class MulticlassConformalCalibrator(MulticlassClassificationCalibrator):
    r"""Create a prediction sets of possible labels :math:`C(X_{\text{test}}) \subset \{1 \mathrel{.\,.} K\}` that follows:

    .. math::
        1 - \alpha \leq \Pr (Y_{\text{test}} \in C(X_{\text{test}})) \leq 1 - \alpha + \frac{1}{n + 1}

    In other words, the probability that the prediction set contains the correct label is almost exactly :math:`1-\alpha`.
    More detailes can be found in [1]_.

    Parameters
    ----------
    alpha: float
        Error rate, :math:`\alpha \in [0, 1]`

    References
    ----------
    .. [1] Angelopoulos, A.N.; Bates, S.; "A Gentle Introduction to Conformal Prediction and Distribution-Free
        Uncertainty Quantification." arXiv Preprint 2021, https://arxiv.org/abs/2107.07511
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"arg `alpha` must be between 0 and 1. got: {alpha}.")

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        r"""Compute nonconformity score as the negative of the softmax output for the true class.

        .. math::
            s_i = -\hat{f}(X_i)_{Y_i}
        """
        return -preds

    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        self.qhats = []
        scores = self.nonconformity_scores(uncs)
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            scores_j = scores[:, j][mask_j]

            scores_j = torch.gather(scores_j, 1, targets_j.unsqueeze(1)).squeeze(1)
            num_data = targets_j.shape[0]
            if self.alpha >= 1 / (num_data + 1):
                q_level = math.ceil((num_data + 1) * (1 - self.alpha)) / num_data
            else:
                q_level = 1
                warnings.warn(
                    "`alpha` is smaller than `1 / (number of data + 1)`, so the `1 - alpha` quantile is set to 1, but this only ensures that the coverage is trivially satisfied."
                )
            qhat = torch.quantile(scores_j, q_level, interpolation="higher")
            self.qhats.append(qhat)

        self.qhats = torch.tensor(self.qhats)
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        calibrated_preds = torch.zeros_like(uncs, dtype=torch.int)
        scores = self.nonconformity_scores(uncs)

        for j, qhat in enumerate(self.qhats):
            calibrated_preds[:, j] = (scores[:, j] <= qhat).int()

        return calibrated_preds


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class AdaptiveMulticlassConformalCalibrator(MulticlassConformalCalibrator):
    @staticmethod
    def nonconformity_scores(preds):
        r"""Compute nonconformity score by greedily including classes in the classification set until it reaches the true label.

        .. math::
            s(x, y) = \sum_{j=1}^{k} \hat{f}(x)_{\pi_j(x)}, \text{ where } y = \pi_k(x)

        where :math:`\pi_k(x)` is the permutation of :math:`\{1 \mathrel{.\,.} K\}` that sorts :math:`\hat{f}(X_{test})` from most likely to least likely.
        """

        sort_index = torch.argsort(-preds, dim=2)
        sorted_preds = torch.gather(preds, 2, sort_index)
        sorted_scores = sorted_preds.cumsum(dim=2)
        unsorted_scores = torch.zeros_like(sorted_scores).scatter_(2, sort_index, sorted_scores)

        return unsorted_scores


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(MulticlassClassificationCalibrator):
    def fit(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Self:
        return self

    def apply(self, uncs: Tensor) -> Tensor:
        return
