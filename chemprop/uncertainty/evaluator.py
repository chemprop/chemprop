from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from chemprop.utils.registry import ClassRegistry

UncertaintyEvaluatorRegistry = ClassRegistry()


class RegressionEvaluator(ABC):
    """Evaluates the quality of uncertainty estimates in regression tasks."""

    @abstractmethod
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

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
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the evaluation

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-regression")
class NLLRegressionEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("miscalibration_area")
class CalibrationAreaEvaluator(RegressionEvaluator):
    """
    A class for evaluating regression uncertainty values based on how they deviate from perfect
    calibration on an observed-probability versus expected-probability plot.
    """

    def evaluate(
        self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor, num_bins: int = 100
    ) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

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
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the evaluation
        num_bins: int
            the number of bins to discretize the ``[0, 1]`` interval

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """
        bins = torch.arange(1, num_bins)
        bin_scaling = torch.special.erfinv(bins / num_bins).view(-1, 1, 1) * np.sqrt(2)
        errors = torch.abs(preds - targets)
        uncs = torch.sqrt(uncs).unsqueeze(0)
        bin_unc = uncs * bin_scaling
        bin_count = bin_unc >= errors.unsqueeze(0)
        mask = mask.unsqueeze(0)
        observed_auc = (bin_count & mask).sum(1) / mask.sum(1)
        num_tasks = uncs.shape[-1]
        observed_auc = torch.cat(
            [torch.zeros(1, num_tasks), observed_auc, torch.ones(1, num_tasks)]
        ).T
        ideal_auc = torch.arange(num_bins + 1) / num_bins
        miscal_area = 0.01 * (observed_auc - ideal_auc).abs().sum(dim=1)
        return miscal_area


@UncertaintyEvaluatorRegistry.register("ence")
class ExpectedNormalizedErrorEvaluator(RegressionEvaluator):
    r"""
    A class that evaluates uncertainty performance by binning together clusters of predictions
    and comparing the average predicted variance of the clusters against the RMSE of the cluster. [1]_

    .. math::
        \text{ENCE} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\mathrm{RMV}_i - \mathrm{RMSE}_i|}{\mathrm{RMV}_i}

    where :math:`N` is the number of bins, :math:`\mathrm{RMV}_i` is the root of the mean uncertainty over the :math:`i`-th bin and :math:`\mathrm{RMSE}_i`
    is the root mean square error over the :math:`i`-th bin.This discrepancy is further normalized by the
    uncertainty overthe bin, :math:`\mathrm{RMV}_i`, because the error is expected to be naturally higher as the uncertainty increases.

    References
    ----------
    .. [1] Levi, D.; Gispan, L.; Giladi, N.; Fetaya, E. "Evaluating and Calibrating Uncertainty Prediction in Regression Tasks." Sensors, 2022, 22(15), 5540. https://www.mdpi.com/1424-8220/22/15/5540.
    """

    def evaluate(
        self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor, num_bins: int = 100
    ) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

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
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the evaluation
        num_bins: int
            the number of bins the data are divided into

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """
        masked_preds = preds * mask
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        errors = torch.abs(masked_preds - masked_targets)

        sort_idx = torch.argsort(masked_uncs, dim=0)
        sorted_uncs = torch.gather(masked_uncs, 0, sort_idx)
        sorted_errors = torch.gather(errors, 0, sort_idx)

        split_unc = torch.chunk(sorted_uncs, num_bins, dim=0)
        split_error = torch.chunk(sorted_errors, num_bins, dim=0)

        root_mean_vars = torch.sqrt(torch.stack([chunk.mean(0) for chunk in split_unc]))
        rmses = torch.sqrt(torch.stack([chunk.pow(2).mean(0) for chunk in split_error]))

        ence = torch.mean(torch.abs(root_mean_vars - rmses) / root_mean_vars, dim=0)
        return ence


@UncertaintyEvaluatorRegistry.register("spearman")
class SpearmanEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-regression")
class ConformalRegressionEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


class BinaryClassificationEvaluator(ABC):
    """Evaluates the quality of uncertainty estimates in binary classification tasks."""

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probability of class 1) of the shape of ``n x t``, where ``n`` is the number of input
            molecules/reactions, and ``t`` is the number of tasks.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the evaluation

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(BinaryClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-classification")
class ConformalMultilabelEvaluator(BinaryClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


class MulticlassClassificationEvaluator(ABC):
    """Evaluates the quality of uncertainty estimates in multiclass classification tasks."""

    @abstractmethod
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """Evaluate the performance of uncertainty predictions against the model target values.

        Parameters
        ----------
        uncs: Tensor
            the predicted uncertainties (i.e., the predicted probabilities for each class) of the shape of ``n x t x c``, where ``n`` is the number of input
            molecules/reactions, ``t`` is the number of tasks, and ``c`` is the number of classes.
        targets: Tensor
            a tensor of the shape ``n x t``
        mask: Tensor
            a tensor of the shape ``n x t`` indicating whether the given values should be used in the evaluation

        Returns
        -------
        Tensor
            a tensor of the shape ``t`` containing the evaluated metrics
        """


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMulticlassEvaluator(MulticlassClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-multiclass")
class ConformalMulticlassEvaluator(MulticlassClassificationEvaluator):
    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return
