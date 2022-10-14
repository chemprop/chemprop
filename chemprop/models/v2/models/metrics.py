from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from torchmetrics import functional as F


class Metric(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class MAEMetric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].abs().mean()


class MSEMetric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return (preds - targets)[mask].square().mean()


class RMSEMetric(MSEMetric):
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
        **kwargs
    ) -> Tensor:
        preds = self.bound_preds(preds, targets, gt_targets, lt_targets)

        return super().__call__(preds, targets, mask)

    def bound_preds(self, preds, targets, gt_targets, lt_targets) -> Tensor:
        preds = torch.where(torch.logical_and(preds < targets, lt_targets), targets, preds)
        preds = torch.where(torch.logical_and(preds > targets, gt_targets), targets, preds)
        
        return preds


class BoundedMAEMetric(BoundedMetric, MAEMetric):
    pass


class BoundedMSEMetric(BoundedMetric, MSEMetric):
    pass


class BoundedRMSEMetric(BoundedMetric, RMSEMetric):
    pass


class R2Metric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.r2_score(preds[mask], targets[mask])


class AUROCMetric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.auroc(preds[mask], targets[mask])


class AUPRCMetric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        p, r, _ = F.precision_recall(preds, targets)

        return F.auc(r, p)


class ThresholdedMetric(Metric):
    def __init__(self, threshold: float = 0.5, **kwargs):
        self.threshold = threshold


class AccuracyMetric(ThresholdedMetric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.accuracy(preds[mask], targets[mask], threshold=self.threshold)


class F1Metric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.f1_score(preds[mask], targets[mask], threshold=self.threshold)


class MCCMetric(ThresholdedMetric):
    def __init__(self, threshold: float = 0.5, n_classes: int = 2, **kwargs) -> Tensor:
        super().__init__(threshold, **kwargs)
        self.n_classes = n_classes

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.matthews_corrcoef(preds[mask], targets[mask], self.n_classes, self.threshold)


class CrossEntropyMetric(Metric):
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return cross_entropy(preds[mask], targets[mask]).mean()

        
class SIDMetric(ThresholdedMetric):
    alias = "spectral-sid"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:        
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (
            torch.log(preds_norm / targets) * preds_norm
            + torch.log(targets / preds_norm) * targets
        )[mask].mean()


class SIDMetric(ThresholdedMetric):
    alias = "spectral-wasserstein"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))[mask].mean()