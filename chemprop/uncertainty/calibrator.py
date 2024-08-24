from abc import abstractmethod

import torch
from torch import Tensor

from chemprop.utils.registry import ClassRegistry


class UncertaintyCalibrator:
    @abstractmethod
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        """
        Fit calibration method for the calibration data.
        """

    @abstractmethod
    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Take in predictions and uncertainty parameters from a model and apply the calibration method using fitted parameters.
        """


UncertaintyCalibratorRegistry = ClassRegistry[UncertaintyCalibrator]()


@UncertaintyCalibratorRegistry.register("zscaling")
class ZScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("tscaling")
class TScalingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("zelikman-interval")
class ZelikmanCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("mve-weighting")
class MVEWeightingCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("platt")
class PlattCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("conformal-multilabel")
class MultilabelConformalCalibrator(UncertaintyCalibrator):
    r"""Creates conformal in-set and conformal out-set such that for :math:`1-\alpha` proportion of datapoints,
    the set of labels is bounded by the in- and out-sets [1]_:
    .. math::
        \mathbb{P}\left(
            \hat{\mathcal C}_{\text{in}}(X) \subseteq \mathcal Y \subseteq \hat{\mathcal C}_{\text{out}}(X)
        \right) \geq 1 - \alpha,

    where the in-set :math:`\hat{\mathcal C}_\text{in}` is contained by the set of true labels :math:` \mathcal Y` and
    :math:` \mathcal Y` is contained within the out-set :math:`\hat{\mathcal C}_\text{out}`.

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
        self.alpha = torch.tensor(alpha)
        if 0 <= self.alpha <= 1:
            raise ValueError(
                f"The error rate (i.e., alpha) should be between 0 and 1. Got {self.alpha}."
            )

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        r"""
        Compute nonconformity score as the negative of the predicted probability.

        .. math::
            s_i = -\hat{f}(X_i)_{Y_i}
        """
        return -preds

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        if targets.shape[1] < 2:
            raise ValueError(
                f"The number of tasks should be laerger than 1. Got {targets.shape[1]}."
            )

        has_zeros = torch.any(targets == 0, dim=1)
        inds_zeros = targets[has_zeros] == 0
        scores_in = self.nonconformity_scores(uncs[has_zeros])
        masked_scores_in = scores_in * inds_zeros.float() + torch.where(
            inds_zeros, torch.zeros_like(scores_in), torch.tensor(float("inf"))
        )
        calibration_scores_in = torch.min(
            masked_scores_in.masked_fill(~mask, float("inf")), dim=1
        ).values

        has_ones = torch.any(targets == 1, dim=1)
        inds_ones = targets[has_ones] == 1
        scores_out = self.nonconformity_scores(uncs[has_ones])
        masked_scores_out = scores_out * inds_ones.float() + torch.where(
            inds_ones, torch.zeros_like(scores_out), torch.tensor(float("-inf"))
        )
        calibration_scores_out = torch.max(
            masked_scores_out.masked_fill(~mask, float("-inf")), dim=1
        ).values

        self.tout = torch.quantile(
            calibration_scores_out, 1 - self.alpha / 2, interpolation="higher"
        )
        self.tin = torch.quantile(calibration_scores_in, self.alpha / 2, interpolation="higher")

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        scores = self.nonconformity_scores(uncs)

        cal_preds_in = (scores <= self.tin).int()
        cal_preds_out = (scores <= self.tout).int()
        cal_preds_in_out = torch.cat((cal_preds_in, cal_preds_out), dim=1)

        return preds, cal_preds_in_out


@UncertaintyCalibratorRegistry.register("conformal-multiclass")
class MulticlassConformalCalibrator(UncertaintyCalibrator):
    r"""Create a prediction sets of possible labels :math:`C(X_{\text{test}}) \subset \{1, \ldots, K\}` that follows:

    .. math::
        1 - \alpha \leq \mathbb{P}(Y_{\text{test}} \in C(X_{\text{test}})) \leq 1 - \alpha + \frac{1}{n + 1}

    In other words, the probability that the prediction set contains the correct label is almost exactly 1 - :math:`\alpha`.
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
        self.alpha = torch.tensor(alpha)
        self.qhats = torch.tensor([])
        if 0 <= self.alpha <= 1:
            raise ValueError(
                f"The error rate (i.e., alpha) should be between 0 and 1. Got {self.alpha}."
            )

    @staticmethod
    def nonconformity_scores(preds: Tensor):
        r"""Compute nonconformity score as the negative of the softmax output for the true class.

        .. math::
            s_i = -\hat{f}(X_i)_{Y_i}
        """
        return -preds

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        scores = self.nonconformity_scores(uncs)
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            scores_j = scores[:, j][mask_j]

            scores_j = torch.gather(scores_j, 1, targets_j.unsqueeze(1)).squeeze(1)
            num_data = targets_j.shape[0]
            q_level = (torch.ceil((num_data + 1) * (1 - self.alpha)) / num_data).item()
            qhat = torch.quantile(scores_j, q_level, interpolation="higher")
            self.qhats = torch.cat((self.qhats, torch.tensor([qhat])))

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        calibrated_preds = torch.zeros_like(uncs, dtype=torch.int)
        scores = self.nonconformity_scores(uncs)

        for j, qhat in enumerate(self.qhats):
            calibrated_preds[:, j] = (scores[:, j] <= qhat).int()

        return preds, calibrated_preds


@UncertaintyCalibratorRegistry.register("conformal-adaptive")
class AdaptiveMulticlassConformalCalibrator(MulticlassConformalCalibrator):
    @staticmethod
    def nonconformity_scores(preds):
        r"""Compute nonconformity score by greedily including classes in the classification set until it reach the true label.

        .. math::
            s(x, y) = \sum_{j=1}^{k} \hat{f}(x)_{\pi_j(x)}, \text{ where } y = \pi_k(x)

        where :math:`\pi_k(x)` is the permutation of :math:`\{1,...,K\}` that sorts :math:`\hat{f}(X_{test})` from most likely to least likely.
        """

        sort_index = torch.argsort(-preds, dim=2)
        sorted_preds = torch.gather(preds, 2, sort_index)
        sorted_scores = sorted_preds.cumsum(dim=2)
        unsorted_scores = torch.zeros_like(sorted_scores).scatter_(2, sort_index, sorted_scores)

        return unsorted_scores


@UncertaintyCalibratorRegistry.register("conformal-regression")
class RegressionConformalCalibrator(UncertaintyCalibrator):
    r"""Conformalize quantiles to make the interval :math:`[\hat{t}_{\alpha/2}(x),\hat{t}_{1-\alpha/2}(x)]` to have
    approximately 1 - :math:`\alpha` coverage. [1]_

    .. math::
        s(x, y) &= \max \left\{ \hat{t}_{\alpha/2}(x) - y, y - \hat{t}_{1-\alpha/2}(x) \right\}

        \hat{q} &= \text{Quantile}(s_1, \ldots, s_n; \left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil)

        C(x) &= \left[ \hat{t}_{\alpha/2}(x) - \hat{q}, \hat{t}_{1-\alpha/2}(x) + \hat{q} \right]

    where :math:`s` is the nonconformity score as the difference between y and its nearest quantile.
    :math:`\hat{t}_{\alpha/2}(x)` and :math:`\hat{t}_{1-\alpha/2}(x)` are the predicted quantiles from quantile
    regression model.

    .. note::
        The algorithm is specifically designed for quantile regression model. Intuitively, the set C(x) just
        grows or shrinks the distance between the quantiles by :math:`\hat{q}` to achieve coverage. However, this
        function can also be applied to regrssion model without quantiles be provided. In this case, both
        :math:`\hat{t}_{\alpha/2}(x)` and :math:`\hat{t}_{1-\alpha/2}(x)` are the same as :math:`\hat{y}`. Then, the
        interval would be the same for every data point (i.e., :math:`\left[-\hat{q}, \hat{q} \right]`).

    Parameters
    ----------
    alpha: float
        The error rate, :math:`\alpha \in [0, 1]`

    References
    ----------
    .. [1] Angelopoulos, A.N.; Bates, S.; "A Gentle Introduction to Conformal Prediction and Distribution-Free
        Uncertainty Quantification." arXiv Preprint 2021, https://arxiv.org/abs/2107.07511
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = torch.tensor(alpha)
        self.qhats = torch.tensor([])
        self.bounds = torch.tensor([-1 / 2, 1 / 2]).view(-1, 1)
        if 0 <= self.alpha <= 1:
            raise ValueError(
                f"The error rate (i.e., alpha) should be between 0 and 1. Got {self.alpha}."
            )

    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        for j in range(preds.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            preds_j = preds[:, j][mask_j]
            interval_j = uncs[:, j][mask_j]

            interval_bounds = self.bounds * interval_j.unsqueeze(0)
            pred_bounds = preds_j.unsqueeze(0) + interval_bounds

            calibration_scores = torch.max(pred_bounds[0] - targets_j, targets_j - pred_bounds[1])

            num_data = targets_j.shape[0]
            q_level = (torch.ceil((num_data + 1) * (1 - self.alpha)) / num_data).item()
            qhat = torch.quantile(calibration_scores, q_level, interpolation="higher")

            self.qhats = torch.cat((self.qhats, torch.tensor([qhat])))

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        cal_intervals = uncs + 2 * self.qhats

        return preds, cal_intervals


@UncertaintyCalibratorRegistry.register("isotonic")
class IsotonicCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return


@UncertaintyCalibratorRegistry.register("isotonic-multiclass")
class IsotonicMulticlassCalibrator(UncertaintyCalibrator):
    def fit(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> None:
        ...
        return

    def apply(self, preds: Tensor, uncs: Tensor) -> tuple[Tensor, Tensor]:
        ...
        return
