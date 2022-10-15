from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.utils.mixins import RegistryMixin


class LossFunction(ABC, RegistryMixin):
    registry = {}

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        """Calculate the *unreduced* loss function value given predicted and target values

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

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


class BoundedMSELoss(MSELoss):
    alias = "regression-bounded"

    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        lt_targets: Tensor,
        gt_targets: Tensor,
        **kwargs,
    ) -> Tensor:
        preds = torch.where(torch.logical_and(preds < targets, lt_targets), targets, preds)
        preds = torch.where(torch.logical_and(preds > targets, gt_targets), targets, preds)

        return super().__call__(preds, targets)


class MVELoss(LossFunction):
    """Calculate the loss using Eq. 9 from [1]_

    References
    ----------
    .. [1] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target probability
    distribution." Proceedings of 1994 IEEE International Conference on Neural Networks, 1994
    https://doi.org/10.1109/icnn.1994.374138
    """

    alias = "regression-mve"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pred_means, pred_vars = preds.split(preds.shape[1] // 2, dim=1)

        A = torch.log(2 * torch.pi * pred_vars) / 2
        B = (pred_means - targets) ** 2 / (2 * pred_vars)

        return A + B


class EvidentialLoss(LossFunction):
    alias = "regression-evidential"

    def __init__(self, v_reg: float = 0, eps: float = 1e-8, **kwargs):
        self.v_reg = v_reg
        self.eps = eps

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
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
    alias = "classification-bce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


class CrossEntropyLoss(LossFunction):
    alias = "multiclass-ce"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        return F.cross_entropy(preds, targets, reduction="none")


class MCCLossBase(LossFunction):
    @abstractmethod
    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, weights: Tensor, **kwargs
    ) -> Tensor:
        pass


class ClassificationMCCLoss(MCCLossBase):
    alias = "classification-mcc"

    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, weights: Tensor, **kwargs
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

        c = (preds * bin_targets * weights * mask).sum()
        s = (preds * weights * mask).sum()
        p_tot = ((preds * weights * mask).sum(0) * (bin_targets * weights * mask).sum(0)).sum()
        p2 = ((preds * weights * mask).sum(0) ** 2).sum()
        t2 = (torch.sum(bin_targets * weights * mask).sum(0) ** 2).sum()

        return 1 - (c * s - p_tot) / torch.sqrt((s**2 - p2) * (s**2 - t2))


class DirichletLossBase(LossFunction):
    """Uses the loss function from [1]_

    References
    ----------
    .. [1] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
    classification uncertainty." Advances in neural information processing systems, 31, 2018. doi:
    https://doi.org/10.48550/arXiv.1806.01768
    """

    def __init__(self, v_kl: float = 0.0, **kwargs):
        self.v_kl = v_kl

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        S = preds.sum(-1, keepdim=True)
        p = preds / S
        A = (targets - p).square().sum(dim=-1, keepdim=True)
        B = torch.sum((p * (1 - p)) / (S + 1), dim=-1, keepdim=True)
        L_sos = A + B

        alpha_hat = targets + (1 - targets) * preds

        beta = torch.ones_like(alpha_hat)
        S_alpha = alpha_hat.sum(-1, keepdim=True)
        S_beta = beta.sum(dim=-1, keepdim=True)

        ln_alpha = S_alpha.lgamma() - alpha_hat.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()

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

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        num_tasks = targets.shape[1]
        num_classes = 2
        preds = preds.reshape(len(preds), num_tasks, num_classes)

        y_one_hot = torch.eye(num_classes, device=preds.device)[targets.long()]

        return super().__call__(preds, y_one_hot)


class DirichletMulticlassLoss(DirichletLossBase):
    alias = "multiclass-dirichlet"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        y_one_hot = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        return super().__call__(preds, y_one_hot)


def build_loss(dataset_type: str, loss_function: str, **kwargs) -> LossFunction:
    key = f"{dataset_type.lower()}-{loss_function.lower()}"

    try:
        return LossFunction.registry[key](**kwargs)
    except KeyError:
        combos = {tuple(k.split("-")) for k in LossFunction.registry.keys()}
        raise ValueError(
            f"dataset type '{dataset_type}' does not support loss function '{loss_function}'! "
            f"Expected one of (`dataset_type`, `loss_function`) combos: {combos}"
        )


class SpectralLoss(LossFunction):
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        self.threshold = threshold

    @abstractmethod
    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        pass


class SIDSpectralLoss(SpectralLoss):
    alias = "spectral-sid"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (
            torch.log(preds_norm / targets) * preds_norm + torch.log(targets / preds_norm) * targets
        )


class WassersteinSpectralLoss(SpectralLoss):
    alias = "spectral-wasserstein"

    def __call__(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))
