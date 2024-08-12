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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j]
            targets_j = targets[:, j]
            uncs_j = uncs[:, j]
            masked_preds = preds_j[mask_j]
            masked_targets = targets_j[mask_j]
            masked_uncs = uncs_j[mask_j]
            errors = masked_preds - masked_targets
            nll = (2 * torch.pi * masked_uncs).log() / 2 + errors**2 / (2 * masked_uncs)
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)


@UncertaintyEvaluatorRegistry.register("nll-classification")
class NLLClassEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j]
            targets_j = targets[:, j]
            uncs_j = uncs[:, j]
            masked_preds = preds_j[mask_j]
            masked_targets = targets_j[mask_j]
            masked_uncs = uncs_j[mask_j]
            likelihood = masked_uncs * masked_targets + (1 - masked_uncs) * (1 - masked_targets)
            nll = -1 * likelihood.log()
            nlls.append(nll.mean(dim=0))
        return torch.stack(nlls)


@UncertaintyEvaluatorRegistry.register("nll-multiclass")
class NLLMultiEvaluator(UncertaintyEvaluator):
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        nlls = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j]
            targets_j = targets[:, j]
            uncs_j = uncs[:, j]
            masked_preds = preds_j[mask_j]
            masked_targets = targets_j[mask_j]
            masked_uncs = uncs_j[mask_j]
            targets_one_hot = torch.eye(masked_uncs.shape[-1])[masked_targets.long()]
            likelihood = (targets_one_hot * masked_uncs).sum(dim=-1)
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
    def evaluate(self, preds: Tensor, uncs: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        spearman_coeffs = []
        for j in range(uncs.shape[1]):
            mask_j = mask[:, j]
            preds_j = preds[:, j]
            targets_j = targets[:, j]
            uncs_j = uncs[:, j]
            masked_preds = preds_j[mask_j]
            masked_targets = targets_j[mask_j]
            masked_uncs = uncs_j[mask_j]
            masked_errs = (masked_preds - masked_targets).abs()
            spearman = SpearmanCorrCoef()
            spearman_coeff = spearman(masked_uncs, masked_errs)
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
