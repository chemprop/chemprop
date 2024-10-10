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

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        bin_scaling = torch.special.erfinv(torch.arange(1, 100) / 100).view(-1, 1, 1) * np.sqrt(2)
        errors = torch.abs(preds - targets)
        uncs = torch.sqrt(uncs).unsqueeze(0)
        bin_unc = uncs * bin_scaling
        bin_count = bin_unc >= errors.unsqueeze(0)
        mask = mask.unsqueeze(0)
        bin_fractions = (bin_count & mask).sum(1) / mask.sum(1)
        num_tasks = uncs.shape[-1]
        bin_fractions = torch.cat(
            [torch.zeros(1, num_tasks), bin_fractions, torch.ones(1, num_tasks)]
        ).T
        miscal_area = torch.sum(0.01 * torch.abs(bin_fractions - torch.arange(101) / 100), dim=1)
        return miscal_area


@UncertaintyEvaluatorRegistry.register("ence")
class ExpectedNormalizedErrorEvaluator(RegressionEvaluator):
    r"""
    A class that evaluates uncertainty performance by binning together clusters of predictions
    and comparing the average predicted variance of the clusters against the RMSE of the cluster. [1]_

    .. math::
        ENCE = \frac{1}{K} \sum_{i=1}^{K} \frac{|\text{mVAR}(i) - RMSE(i)|}{\text{mVAR}(i)}

    where :math:`mVAR(i)` is the root of the mean uncertainty over the :math:`i`-th bin and :math:`RMSE(i)`
    is the root mean square error over the :math:`i`-th bin.This discrepancy is further normalized by the
    uncertainty overthe bin, :math:`mVAR(i)`, because the error is expected to be naturally higher as the uncertainty increases.

    References
    ----------
    .. [1] Scalia, G.; Grambow, C. A.; Pernici, B.; Li, Y. P.; Green, W. H. "Evaluating Scalable Uncertainty Estimation Methods for Deep Learning-Based Molecular Property Prediction." J. Chem. Inf. Model., 2020, 60(6), 2697-2717. https://pubs.acs.org/doi/10.1021/acs.jcim.9b00975.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        root_mean_vars = torch.zeros([uncs.shape[1], 100])
        rmses = torch.zeros_like(root_mean_vars)

        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            errors = torch.abs(preds_j - targets_j)

            sort_idx = torch.argsort(uncs_j)
            uncs_j = uncs_j[sort_idx]
            errors = errors[sort_idx]

            split_unc = torch.chunk(uncs_j, 100)
            split_error = torch.chunk(errors, 100)

            root_mean_vars[j] = torch.tensor([torch.sqrt(torch.mean(chunk)) for chunk in split_unc])
            rmses[j] = torch.tensor([torch.sqrt(torch.mean(chunk**2)) for chunk in split_error])

        ence = torch.mean(torch.abs(root_mean_vars - rmses) / root_mean_vars, axis=1)
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
