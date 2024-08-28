from abc import abstractmethod

import torch
from torch import Tensor

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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMultiEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        ...
        return


@UncertaintyEvaluatorRegistry.register("conformal-coverage-regression")
class RegressionConformalEvaluator(UncertaintyEvaluator):
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


@UncertaintyEvaluatorRegistry.register("conformal-coverage-multiclass")
class MulticlassConformalEvaluator(UncertaintyEvaluator):
    r"""
    Evaluate the coverage of conformal prediction for multiclass classification dataset.

    .. math::
        \Pr (Y_{\text{test}} \in C(X_{\text{test}}))

    where the :math:`C(X_{\text{test}})) \subset \{1 \mathrel{.\,.} K\}` is a prediction set of possible labels .
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=uncs.shape[2])
        covered_mask = torch.max(uncs * targets_one_hot, dim=-1)[0] > 0
        return (covered_mask & mask).sum(0) / mask.sum(0)


@UncertaintyEvaluatorRegistry.register("conformal-coverage-classification")
class MultilabelConformalEvaluator(UncertaintyEvaluator):
    r"""
    Evaluate the coverage of conformal prediction for binary classification dataset with multiple labels.

    .. math::
        \Pr \left(
            \hat{\mathcal C}_{\text{in}}(X) \subseteq \mathcal Y \subseteq \hat{\mathcal C}_{\text{out}}(X)
        \right)

    where the in-set :math:`\hat{\mathcal C}_\text{in}` is contained by the set of true labels :math:`\mathcal Y` and
    :math:`\mathcal Y` is contained within the out-set :math:`\hat{\mathcal C}_\text{out}`.
    """

    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        in_set, out_set = torch.chunk(uncs, 2, 1)
        covered_mask = torch.logical_and(in_set <= targets, targets <= out_set)
        return (covered_mask & mask).sum(0) / mask.sum(0)
