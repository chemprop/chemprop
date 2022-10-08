from abc import abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.utils.mixins import RegistryMixin


class LossFunction(Callable, RegistryMixin):
    registry = {}

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Calculate the loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a tensor of shape `b x ...` containing the raw model predictions
        targets : Tensor
            a tensor of shape `b x ...` containing the target values
        **kwargs
            keyword arguments specific to the given loss function

        Returns
        -------
        Tensor
            a tensor of shape `b x ...` containing the loss value for each prediction
        """
        pass


class MSELoss(LossFunction):
    alias = "regression-mse"

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


class BoundedMSELoss(MSELoss):
    alias = "regression-bounded"

    def __call__(
        self, preds: Tensor, targets: Tensor, lt_targets: Tensor, gt_targets: Tensor, **kwargs
    ) -> Tensor:
        preds = torch.where(torch.logical_and(preds < targets, lt_targets), targets, preds)
        preds = torch.where(torch.logical_and(preds > targets, gt_targets), targets, preds,)

        return super().__call__(preds, targets)


class MVELoss(LossFunction):
    alias = "regression-mve"

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        pred_means, pred_vars = preds.split(preds.shape[1] // 2, dim=1)

        return (
            torch.log(2 * torch.pi * pred_vars) / 2 + (pred_means - targets) ** 2 / (2 * pred_vars)
        )


class EvidentialLoss(LossFunction):
    alias = "regression-evidential"
    
    def __init__(self, v_reg: float = 0, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.v_reg = v_reg
        self.eps = eps

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        mu, v, alpha, beta = preds.split(preds.shape[1] // 4, dim=1)

        twoBlambda = 2 * beta * (1 + v)
        L_nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        L_reg = (2 * v + alpha) * (targets - mu).abs()

        return L_nll + self.v_reg * (L_reg - self.eps)


class BCELoss(LossFunction):
    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


class CrossEntropyLoss(LossFunction):
    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.cross_entropy(preds, targets, reduction="none")

        
class MCCLossBase(LossFunction):
    @abstractmethod
    def __call__(
        self, preds: Tensor, targets: Tensor, weights: Tensor, mask: Tensor, **kwargs
    ) -> Tensor:
        pass


class ClassificationMCCLoss(MCCLossBase):
    alias = "classification-mcc"

    def __call__(
        self, preds: Tensor, targets: Tensor, weights: Tensor, mask: Tensor, **kwargs
    ) -> Tensor:
        TP = (targets * preds * weights * mask).sum(0)
        FP = ((1 - targets) * preds * weights * mask).sum(0)
        TN = ((1 - targets) * (1 - preds) * weights * mask).sum(0)
        FN = (targets * (1 - preds) * weights * mask).sum(0)
        
        return 1 - ((TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))).sqrt()


class MulticlassMCCLoss(MCCLossBase):
    alias = "multiclass-mcc"

    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, weights: Tensor, **kwargs
    ) -> Tensor:
        mask = mask.unsqueeze(1)
        bin_targets = torch.zeros_like(preds, device=preds.device)
        bin_targets[range(len(preds)), targets] = 1

        c = torch.sum(preds * bin_targets * weights * mask)
        s = torch.sum(preds * weights * mask)
        pt = torch.sum(
            (preds * weights * mask).sum(0) * (bin_targets * weights * mask).sum(0)
        )
        p2 = torch.sum((preds * weights * mask).sum(0) ** 2)
        t2 = torch.sum(torch.sum(bin_targets * weights * mask).sum(0) ** 2)
        
        return 1 - (c * s - pt) / ((s ** 2 - p2) * (s ** 2 - t2)).sqrt()


class SpectralLoss(LossFunction):
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class SIDSpectralLoss(SpectralLoss):
    alias = "spectral-sid"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        device = preds.device
        zero_sub = torch.zeros_like(preds, device=device)
        one_sub = torch.ones_like(preds, device=device)
        
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds = preds.masked_scatter(~mask, zero_sub)
        preds_sum = torch.sum(preds, axis=1, keepdim=True)
        preds_norm = torch.div(preds, preds_sum)

        targets = targets.masked_scatter(~mask, one_sub)
        preds_norm = preds_norm.masked_scatter(~mask, one_sub)

        return (
            preds_norm.div(targets).log().mul(preds_norm) 
            + targets.div(preds_norm).log().mul(targets)
        )


class WassersteinSpectralLoss(SpectralLoss):
    alias = "spectral-wasserstein"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        device = preds.device
        zero_sub = torch.zeros_like(preds, device=device)

        if self.threshold is not None:
            preds = preds.clamp(self.threshold)

        preds = preds.masked_scatter(~mask, zero_sub)
        preds_sum = torch.sum(preds, axis=1, keepdim=True)
        preds_norm = torch.div(preds, preds_sum)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))


class DirichletLossBase(LossFunction):
    def __init__(self, v_kl: float = 0., **kwargs):
        super().__init__(**kwargs)
        self.v_kl = v_kl

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        S = torch.sum(preds, dim=-1, keepdim=True)
        p = preds / S
        A = torch.sum((targets - p) ** 2, dim=-1, keepdim=True)
        B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
        L_sos = A + B

        alpha_hat = targets + (1 - targets) * preds

        beta = torch.ones_like(alpha_hat)
        S_alpha = torch.sum(alpha_hat, dim=-1, keepdim=True)
        S_beta = torch.sum(beta, dim=-1, keepdim=True)

        ln_alpha = (
            torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha_hat), dim=-1, keepdim=True)
        )
        ln_beta = (
            torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(S_beta)
        )

        dg_alpha = torch.digamma(alpha_hat)
        dg_S_alpha = torch.digamma(S_alpha)

        L_kl = (
            ln_alpha
            + ln_beta
            + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), dim=-1, keepdim=True)
        )

        return (L_sos + self.v_kl * L_kl).mean(-1)
    

class DirichletClassificationLoss(DirichletLossBase):
    alias = "classification-dirichlet"

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        num_tasks = targets.shape[1]
        num_classes = 2
        preds = preds.reshape(len(preds), num_tasks, num_classes)

        y_one_hot = torch.eye(num_classes, device=preds.device)[targets.long()]

        return super().__call__(preds, y_one_hot)


class DirichletMulticlassLoss(DirichletLossBase):
    alias = "multiclass-dirichlet"

    def __call__(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        y_one_hot = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        return super().__call__(preds, y_one_hot)


def get_loss(dataset_type: str, loss_function: str, **kwargs) -> LossFunction:
    key = f"{dataset_type.lower()}-{loss_function.lower()}"

    try:
        return LossFunction.registry[key](**kwargs)
    except KeyError:
        combos = {tuple(k.split("-")) for k in LossFunction.registry.keys()}
        raise ValueError(
            f"dataset type '{dataset_type}' does not support loss function '{loss_function}'! "
            f"Expected one of (`daaset_type`, `loss_function`) combos: {combos}"
        )