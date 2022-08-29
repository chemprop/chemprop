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
    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        """Calculate the loss function value given predicted and target values

        Parameters
        ----------
        Y_pred : Tensor
            a tensor of shape `b x ...` containing the raw model predictions
        Y_target : Tensor
            a tensor of shape `b x ...` containing the target values

        Returns
        -------
        Tensor
            a tensor of shape `b x ...` containing the loss value for each prediction
        """
        pass


class MSELoss(LossFunction):
    alias = "regression-mse"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        return (Y_pred - Y_target) ** 2


class BoundedMSELoss(MSELoss):
    alias = "regression-bounded"

    def __call__(
        self, Y_pred: Tensor, Y_target: Tensor, lt_targets: Tensor, gt_targets: Tensor, **kwargs
    ) -> Tensor:
        Y_pred = torch.where(
            torch.logical_and(Y_pred < Y_target, lt_targets), Y_target, Y_pred
        )

        Y_pred = torch.where(
            torch.logical_and(Y_pred > Y_target, gt_targets), Y_target, Y_pred,
        )

        return super().__call__(Y_pred, Y_target)


class MVELoss(LossFunction):
    alias = "regression-mve"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        Y_pred_mean, Y_pred_var = Y_pred.split(Y_pred.shape[1] // 2, dim=1)

        return (
            torch.log(2 * torch.pi * Y_pred_var) / 2
            + (Y_pred_mean - Y_target) ** 2 / (2 * Y_pred_var)
        )


class EvidentialLoss(LossFunction):
    alias = "regression-evidential"
    
    def __init__(self, v_reg: float = 0, eps: float = 1e-8, **kwargs):
        self.v_reg = v_reg
        self.eps = eps

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        mu, v, alpha, beta = Y_pred.split(Y_pred.shape[1] // 4, dim=1)

        twoBlambda = 2 * beta * (1 + v)
        L_nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(twoBlambda)
            + (alpha + 0.5) * torch.log(v * (Y_target - mu) ** 2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        L_reg = (2 * v + alpha) * (Y_target - mu).abs()

        return L_nll + self.v_reg * (L_reg - self.eps)


class BCELoss(LossFunction):
    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        return F.binary_cross_entropy_with_logits(Y_pred, Y_target, reduction="none")


class CrossEntropyLoss(LossFunction):
    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        return F.cross_entropy(Y_pred, Y_target, reduction="none")

        
class MCCLossBase(LossFunction):
    @abstractmethod
    def __call__(self, Y_pred: Tensor, Y_target: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
        pass


class ClassificationMCCLoss(MCCLossBase):
    alias = "classification-mcc"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
        TP = (Y_target * Y_pred * weights * mask).sum(0)
        FP = ((1 - Y_target) * Y_pred * weights * mask).sum(0)
        FN = (Y_target * (1 - Y_pred) * weights * mask).sum(0)
        TN = ((1 - Y_target) * (1 - Y_pred) * weights * mask).sum(0)
        
        return 1 - ((TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))).sqrt()


class MulticlassMCCLoss(MCCLossBase):
    alias = "multiclass-mcc"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
        mask = mask.unsqueeze(1)
        bin_targets = torch.zeros_like(Y_pred, device=Y_pred.device)
        bin_targets[range(len(Y_pred)), Y_target] = 1

        c = torch.sum(Y_pred * bin_targets * weights * mask)
        s = torch.sum(Y_pred * weights * mask)
        pt = torch.sum(
            (Y_pred * weights * mask).sum(0) * (bin_targets * weights * mask).sum(0)
        )
        p2 = torch.sum((Y_pred * weights * mask).sum(0) ** 2)
        t2 = torch.sum(torch.sum(bin_targets * weights * mask).sum(0) ** 2)
        
        return 1 - (c * s - pt) / ((s ** 2 - p2) * (s ** 2 - t2)).sqrt()


class SpectralLoss(LossFunction):
    @abstractmethod
    def __call__(self, Y_pred: Tensor, Y_target: Tensor, mask: Tensor, threshold: Optional[float] = None) -> Tensor:
        pass


class SIDSpectralLoss(SpectralLoss):
    alias = "spectral-sid"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, mask: Tensor, threshold: Optional[float] = None) -> Tensor:
        device = Y_pred.device
        zero_sub = torch.zeros_like(Y_pred, device=device)
        one_sub = torch.ones_like(Y_pred, device=device)
        
        if threshold is not None:
            Y_pred = Y_pred.clamp(threshold)

        Y_pred = torch.where(mask, Y_pred, zero_sub)
        Y_pred_sum = torch.sum(Y_pred, axis=1, keepdim=True)
        Y_pred_norm = torch.div(Y_pred, Y_pred_sum)

        Y_target = torch.where(mask, Y_target, one_sub)
        Y_pred_norm = torch.where(mask, Y_pred_norm, one_sub)

        return (
            Y_pred_norm.div(Y_target).log().mul(Y_pred_norm) 
            + Y_target.div(Y_pred_norm).log().mul(Y_target)
        )


class WassersteinSpectralLoss(SpectralLoss):
    alias = "spectral-wasserstein"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, mask: Tensor, threshold: Optional[float] = None) -> Tensor:
        device = Y_pred.device
        zero_sub = torch.zeros_like(Y_pred, device=device)

        if threshold is not None:
            Y_pred = Y_pred.clamp(threshold)

        Y_pred = torch.where(mask, Y_pred, zero_sub)
        Y_pred_sum = torch.sum(Y_pred, axis=1, keepdim=True)
        Y_pred_norm = torch.div(Y_pred, Y_pred_sum)

        return torch.abs(Y_target.cumsum(1) - Y_pred_norm.cumsum(1))


class DirichletLossBase(LossFunction):
    def __init__(self, v_kl: float = 0.):
        self.v_kl = v_kl

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        S = torch.sum(Y_pred, dim=-1, keepdim=True)
        p = Y_pred / S
        A = torch.sum((Y_target - p) ** 2, dim=-1, keepdim=True)
        B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
        L_sos = A + B

        alpha_hat = Y_target + (1 - Y_target) * Y_pred

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

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        num_tasks = Y_target.shape[1]
        num_classes = 2
        Y_pred = Y_pred.reshape(len(Y_pred), num_tasks, num_classes)

        y_one_hot = torch.eye(num_classes, device=Y_pred.device)[Y_target.long()]

        return super().__call__(Y_pred, y_one_hot)


class DirichletMulticlassLoss(DirichletLossBase):
    alias = "multiclass-dirichlet"

    def __call__(self, Y_pred: Tensor, Y_target: Tensor, **kwargs) -> Tensor:
        y_one_hot = torch.eye(Y_pred.shape[2], device=Y_pred.device)[Y_target.long()]

        return super().__call__(Y_pred, y_one_hot)


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