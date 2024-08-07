from abc import abstractmethod

from torch import Tensor
import torch
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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        masked_preds = preds * mask
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        nlls = (2 * torch.pi * masked_uncs).log() / 2 + (masked_preds - masked_targets) ** 2 / (2 * masked_uncs)
        return nlls.mean(dim = 0)


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        likelihoods = masked_uncs * masked_targets + (1 - masked_uncs) * (1 - masked_targets)
        nlls = -1 * likelihoods.log()
        return nlls.mean(dim = 0)


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMultiEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        targets_shape = torch.nn.functional.one_hot(masked_targets, masked_uncs.shape[-1])
        likelihoods = (targets_shape * masked_uncs).sum(dim = -1)
        nlls = -1 * likelihoods.log()
        return nlls.mean(dim = 0)


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
        masked_preds = preds * mask
        masked_targets = targets * mask
        masked_uncs = uncs * mask
        masked_errs = (masked_preds - masked_targets).abs()
        spearman = SpearmanCorrCoef(num_outputs = masked_targets.shape[1])
        return spearman(masked_uncs, masked_errs).unsqueeze(0)


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
