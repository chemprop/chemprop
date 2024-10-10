from abc import ABC, abstractmethod

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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("ence")
class ExpectedNormalizedErrorEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("spearman")
class SpearmanEvaluator(RegressionEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-regression")
class RegressionConformalEvaluator(RegressionEvaluator):
    r"""
    Evaluate the coverage of conformal prediction for regression dataset.

    .. math::
        \Pr (Y_{\text{test}} \in C(X_{\text{test}}))

    where the :math:`C(X_{\text{test}}))` is the predicted interval.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        bounds = torch.tensor([-1 / 2, 1 / 2], device=mask.device)
        interval = uncs.unsqueeze(0) * bounds.view([-1] + [1] * preds.ndim)
        lower, upper = preds.unsqueeze(0) + interval
        covered_mask = torch.logical_and(lower <= targets, targets <= upper)

        return (covered_mask & mask).sum(0) / mask.sum(0)


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
class MultilabelConformalEvaluator(BinaryClassificationEvaluator):
    r"""
    Evaluate the coverage of conformal prediction for binary classification dataset with multiple labels.

    .. math::
        \Pr \left(
            \hat{\mathcal C}_{\text{in}}(X) \subseteq \mathcal Y \subseteq \hat{\mathcal C}_{\text{out}}(X)
        \right)

    where the in-set :math:`\hat{\mathcal C}_\text{in}` is contained by the set of true labels :math:`\mathcal Y` and
    :math:`\mathcal Y` is contained within the out-set :math:`\hat{\mathcal C}_\text{out}`.
    """

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        in_set, out_set = torch.chunk(uncs, 2, 1)
        covered_mask = torch.logical_and(in_set <= targets, targets <= out_set)
        return (covered_mask & mask).sum(0) / mask.sum(0)


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
class MulticlassConformalEvaluator(MulticlassClassificationEvaluator):
    r"""
    Evaluate the coverage of conformal prediction for multiclass classification dataset.

    .. math::
        \Pr (Y_{\text{test}} \in C(X_{\text{test}}))

    where the :math:`C(X_{\text{test}})) \subset \{1 \mathrel{.\,.} K\}` is a prediction set of possible labels .
    """

    def evaluate(self, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=uncs.shape[2])
        covered_mask = torch.max(uncs * targets_one_hot, dim=-1)[0] > 0
        return (covered_mask & mask).sum(0) / mask.sum(0)
