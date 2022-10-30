from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchmetrics import functional as F

from chemprop.v2.utils.mixins import RegistryMixin, FactoryMixin


class Metric(ABC, RegistryMixin):
    """A `Metric` is like a loss function, but it calculates only a single scalar for the entire
    batch.

    NOTE(degraff): this can probably be rewritten to subclass from `LossFunction`
    """

    registry = {}

    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def minimize(cls) -> bool:
        "whether this metric should be minimized"

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class MinimizedMetric(Metric):
    @classmethod
    @property
    def minimize(cls) -> bool:
        return True


class MaximizedMetric(Metric):
    @classmethod
    @property
    def minimize(cls) -> bool:
        return False


class ThresholdedMixin:
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold


class MAEMetric(MinimizedMetric):
    alias = "mae"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].abs().mean()


class MSEMetric(MinimizedMetric):
    alias = "mse"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].square().mean()


class RMSEMetric(MSEMetric):
    alias = "rmse"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return super().__call__(preds, targets, mask).sqrt()


class BoundedMetric(Metric):
    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        gt_targets: Tensor,
        lt_targets: Tensor,
        **kwargs,
    ) -> Tensor:
        preds = self.bound_preds(preds, targets, gt_targets, lt_targets)

        return super().__call__(preds, targets, mask)

    def bound_preds(self, preds, targets, gt_targets, lt_targets) -> Tensor:
        preds = torch.where(torch.logical_and(preds < targets, lt_targets), targets, preds)
        preds = torch.where(torch.logical_and(preds > targets, gt_targets), targets, preds)

        return preds


class BoundedMAEMetric(BoundedMetric, MAEMetric):
    alias = "bounded-mse"


class BoundedMSEMetric(BoundedMetric, MSEMetric):
    alias = "bounded-rmse"


class BoundedRMSEMetric(BoundedMetric, RMSEMetric):
    alias = "bounded-mae"


class R2Metric(MaximizedMetric):
    alias = "r2"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.r2_score(preds[mask], targets[mask])


class AUROCMetric(MaximizedMetric):
    alias = "roc"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.auroc(preds[mask], targets[mask].long())


class AUPRCMetric(MaximizedMetric):
    alias = "prc"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        p, r, _ = F.precision_recall(preds, targets.long())

        return F.auc(r, p)


class AccuracyMetric(MaximizedMetric, ThresholdedMixin):
    alias = "accuracy"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.accuracy(preds[mask], targets[mask].long(), threshold=self.threshold)


class F1Metric(MaximizedMetric):
    alias = "f1"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.f1_score(preds[mask], targets[mask].long(), threshold=self.threshold)


class BCEMetric(MaximizedMetric):
    alias = "bce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return binary_cross_entropy_with_logits(preds[mask], targets[mask].long())


class MCCMetric(MaximizedMetric):
    alias = "mcc"

    def __init__(self, threshold: float = 0.5, n_classes: int = 2, **kwargs) -> Tensor:
        self.threshold = threshold
        self.n_classes = n_classes

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.matthews_corrcoef(
            preds[mask], targets[mask].long(), self.n_classes, self.threshold
        )


class CrossEntropyMetric(MinimizedMetric):
    alias = "ce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return cross_entropy(preds[mask], targets[mask].long())


class SIDMetric(MinimizedMetric, ThresholdedMixin):
    alias = "sid"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (
            torch.log(preds_norm / targets) * preds_norm + torch.log(targets / preds_norm) * targets
        )[mask].mean()


class WassersteinMetric(MinimizedMetric, ThresholdedMixin):
    alias = "wasserstein"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))[mask].mean()


class MetricFactory(Metric, FactoryMixin):
    pass