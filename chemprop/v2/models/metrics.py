from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchmetrics import functional as F

from chemprop.v2.utils import RegistryMixin


class Metric(ABC, RegistryMixin):
    """A `Metric` is like a loss function, but it calculates only a single scalar for the entire
    batch.

    NOTE(degraff): this can probably be rewritten to subclass from `LossFunction`
    """

    registry = {}

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class MAEMetric(Metric):
    alias = "mae"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].abs().mean()


class MSEMetric(Metric):
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


class R2Metric(Metric):
    alias = "r2"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.r2_score(preds[mask], targets[mask])


class AUROCMetric(Metric):
    alias = "roc"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.auroc(preds[mask], targets[mask])


class AUPRCMetric(Metric):
    alias = "prc"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        p, r, _ = F.precision_recall(preds, targets)

        return F.auc(r, p)


class ThresholdedMetric(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold


class AccuracyMetric(ThresholdedMetric):
    alias = "accuracy"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.accuracy(preds[mask], targets[mask], threshold=self.threshold)


class F1Metric(Metric):
    alias = "f1"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.f1_score(preds[mask], targets[mask], threshold=self.threshold)


class BCEMetric(Metric):
    alias = "bce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return binary_cross_entropy_with_logits(preds[mask], targets[mask])


class MCCMetric(ThresholdedMetric):
    alias = "mcc"

    def __init__(self, threshold: float = 0.5, n_classes: int = 2, **kwargs) -> Tensor:
        super().__init__(threshold, **kwargs)
        self.n_classes = n_classes

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.matthews_corrcoef(
            preds[mask], targets[mask].long(), self.n_classes, self.threshold
        )


class CrossEntropyMetric(Metric):
    alias = "ce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return cross_entropy(preds[mask], targets[mask].long())


class SIDMetric(ThresholdedMetric):
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


class WassersteinMetric(ThresholdedMetric):
    alias = "wasserstein"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))[mask].mean()
