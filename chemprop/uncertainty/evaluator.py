from abc import abstractmethod

import torch
from torch import Tensor
from torchmetrics.regression import SpearmanCorrCoef

from chemprop.utils.registry import ClassRegistry


class UncertaintyEvaluator:
    """
    A class for evaluating the effectiveness of uncertainty estimates with metrics.
    """

    @abstractmethod
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        """
        Evaluate the performance of uncertainty predictions against the model target values.
        """


UncertaintyEvaluatorRegistry = ClassRegistry[UncertaintyEvaluator]()


class MetricEvaluator(UncertaintyEvaluator):
    """
    A class for evaluating confidence estimates of classification and multiclass datasets using builtin evaluation metrics.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("nll-regression")
class NLLRegressionEvaluator(UncertaintyEvaluator):
    r"""
    Evaluate uncertainty values for regression datasets using the mean negative-log-likelihood
    of the targets given the probability distributions estimated by the model:

    .. math::

        \mathrm{NLL}(y, \hat y) = \frac{1}{2} \log(2 \pi \sigma^2) + \frac{(y - \hat{y})^2}{2 \sigma^2},

    where :math:`\hat{y}` is the predicted value, :math:`y` is the true value, and
    :math:`\sigma^2` is the predicted uncertainty (variance).

    The function returns a tensor containing the mean NLL for each task.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            errors = preds_j - targets_j
            nll = (2 * torch.pi * uncs_j).log() / 2 + errors**2 / (2 * uncs_j)
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(UncertaintyEvaluator):
    """
    Evaluate uncertainty values for binary classification datasets using the mean negative-log-likelihood
    of the targets given the assigned probabilities from the model:

    .. math::

        \mathrm{NLL} = -\log(\hat{y} \cdot y + (1 - \hat{y}) \cdot (1 - y))

    where :math:`y` is the true binary label (0 or 1), and
    :math:`\hat{y}` is the predicted probability associated with the class label `1`.

    The function returns a tensor containing the mean NLL for each task.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            likelihood = uncs_j * targets_j + (1 - uncs_j) * (1 - targets_j)
            nll = -1 * likelihood.log()
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMultiEvaluator(UncertaintyEvaluator):
    """
    Evaluate uncertainty values for multiclass classification datasets using the mean negative-log-likelihood
    of the targets given the assigned probabilities from the model:

    .. math::

        \mathrm{NLL} = -\log(p_{y_i})

    where :math:`p_{y_i}` is the predicted probability for the true class :math:`y_i`, calculated as:

    .. math::

        p_{y_i} = \sum_{k=1}^{K} \mathbb{1}(y_i = k) \cdot p_k

    Here: :math:`K` is the total number of classes,
    :math:`\mathbb{1}(y_i = k)` is the indicator function that is 1 when the true class :math:`y_i` equals class :math:`k`, and 0 otherwise,
    and :math:`p_k` is the predicted probability for class :math:`k`.

    The function returns a tensor containing the mean NLL for each task.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            targets_one_hot = torch.eye(uncs_j.shape[-1])[targets_j.long()]
            likelihood = (targets_one_hot * uncs_j).sum(dim=-1)
            nll = -1 * likelihood.log()
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)


@UncertaintyEvaluatorRegistry.register("miscalibration_area")
class CalibrationAreaEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("ence")
class ExpectedNormalizedErrorEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("spearman")
class SpearmanEvaluator(UncertaintyEvaluator):
    """
    Evaluate the Spearman rank correlation coefficient between the uncertainties and errors in the model predictions.

    The correlation coefficient returns a value in the [-1, 1] range, with better scores closer to 1
    observed when the uncertainty values are predictive of the rank ordering of the errors in the model prediction.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        spearman_coeffs = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j][mask_j]
            targets_j = targets[:, j][mask_j]
            uncs_j = uncs[:, j][mask_j]
            errs_j = (preds_j - targets_j).abs()
            spearman = SpearmanCorrCoef()
            spearman_coeff = spearman(uncs_j, errs_j)
            spearman_coeffs.append(spearman_coeff)
        return torch.stack(spearman_coeffs)


@UncertaintyEvaluatorRegistry.register("conformal-coverage-regression")
class ConformalRegressionEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-multiclass")
class ConformalMulticlassEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-classification")
class ConformalMultilabelEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return
