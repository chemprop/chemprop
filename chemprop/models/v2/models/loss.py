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

    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor, w_t: Tensor, **kwargs
    ):
        """Calculate the *reduced* loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a float tensor of shape `b x t x ...` containing the raw model predictions
        targets : Tensor
            a float tensor of shape `b x t` containing the target values
        mask : Tensor
            a boolean tensor of shape `b x t` indicating whether the given sample should be included
            in the loss calculation
        w_d : Tensor
            a tensor of shape `b x 1` containing the per-sample weight
        w_t : Tensor
            a tensor of shape `1 x t` containing the per-task weight
        **kwargs
            keyword arguments specific to the given loss function

        Returns
        -------
        Tensor
            a scalar containing the loss
        """
        L = self.calc(preds, targets, mask=mask, **kwargs)
        L = L * w_d * w_t * mask

        return L.sum() / mask.sum()

    @abstractmethod
    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Calculate the *un*reduced loss function given predicted and target values"""


class MSELoss(LossFunction):
    alias = "regression-mse"

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


class BoundedMSELoss(MSELoss):
    alias = "regression-bounded"

    def calc(
        self, preds: Tensor, targets: Tensor, lt_targets: Tensor, gt_targets: Tensor, **kwargs
    ) -> Tensor:
        preds = torch.where(torch.logical_and(preds < targets, lt_targets), targets, preds)
        preds = torch.where(torch.logical_and(preds > targets, gt_targets), targets, preds)

        return super().calc(preds, targets)


class MVELoss(LossFunction):
    """Calculate the loss using Eq. 9 from [1]_

    References
    ----------
    .. [1] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target probability
    distribution." Proceedings of 1994 IEEE International Conference on Neural Networks, 1994
    https://doi.org/10.1109/icnn.1994.374138
    """

    alias = "regression-mve"

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        pred_means, pred_vars = preds.split(preds.shape[1] // 2, dim=1)

        L_sos = (pred_means - targets) ** 2 / (2 * pred_vars)
        L_kl = (2 * torch.pi * pred_vars).log() / 2

        return L_sos + L_kl


class EvidentialLoss(LossFunction):
    """
    References
    ----------
    .. [1] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.; "Evidential
    Deep Learning for Guided Molecular Property Prediction and Discovery." ACS Cent. Sci. 2021, 7,
    8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    alias = "regression-evidential"

    def __init__(self, v_reg: float = 0.2, eps: float = 1e-8, **kwargs):
        self.v_reg = v_reg
        self.eps = eps

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
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

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


class CrossEntropyLoss(LossFunction):
    alias = "multiclass-ce"

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()

        return F.cross_entropy(preds, targets, reduction="none")


class MCCLossBase(LossFunction):
    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor, w_t: Tensor, **kwargs
    ):
        if not (0 <= preds.min() and preds.max() <= 1):  # transform logits
            preds = preds.softmax(2)

        L = self.calc(preds, targets.long(), mask=mask, w_d=w_d, **kwargs)
        L = L * w_t

        return L.mean()


class ClassificationMCCLoss(MCCLossBase):
    """Calculate a soft Matthews correlation coefficient loss for binary classification

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Phi_coefficient
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    alias = "classification-mcc"

    def calc(self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor) -> Tensor:
        TP = (targets * preds * w_d * mask).sum(0, keepdim=True)
        FP = ((1 - targets) * preds * w_d * mask).sum(0, keepdim=True)
        TN = ((1 - targets) * (1 - preds) * w_d * mask).sum(0, keepdim=True)
        FN = (targets * (1 - preds) * w_d * mask).sum(0, keepdim=True)

        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).sqrt()

        return 1 - MCC


class MulticlassMCCLoss(MCCLossBase):
    """Calculate a soft Matthews correlation coefficient loss for multiclass classification

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    alias = "multiclass-mcc"

    def calc(self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor) -> Tensor:
        device = preds.device

        C = preds.shape[2]
        bin_targets = torch.eye(C, device=device)[targets]
        bin_preds = torch.eye(C, device=device)[preds.argmax(-1)]
        masked_data_weights = w_d.unsqueeze(2) * mask.unsqueeze(2)

        p = (bin_preds * masked_data_weights).sum(0)
        t = (bin_targets * masked_data_weights).sum(0)
        c = (bin_preds * bin_targets * masked_data_weights).sum((0, 2))
        s = (preds * masked_data_weights).sum((0, 2))
        s2 = s.square()

        # the `einsum` calls amount to calculating the batched dot product
        cov_ytyp = c * s - torch.einsum("ij,ij->i", p, t)
        cov_ypyp = s2 - torch.einsum("ij,ij->i", p, p)
        cov_ytyt = s2 - torch.einsum("ij,ij->i", t, t)

        x = cov_ypyp * cov_ytyt
        if x == 0:
            MCC = torch.tensor(0.0, device=device)
        else:
            MCC = cov_ytyp / x.sqrt()

        return 1 - MCC


class DirichletLossBase(LossFunction):
    """Uses the loss function from [1]_

    References
    ----------
    .. [1] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
    classification uncertainty." Advances in neural information processing systems, 31, 2018.
    https://doi.org/10.48550/arXiv.1806.01768
    """

    def __init__(self, v_kl: float = 1.0, **kwargs):
        self.v_kl = v_kl

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        S = preds.sum(-1, keepdim=True)
        p = preds / S
        A = (targets - p).square().sum(-1, keepdim=True)
        B = ((p * (1 - p)) / (S + 1)).sum(-1, keepdim=True)
        L_sos = A + B

        alpha_hat = targets + (1 - targets) * preds

        beta = torch.ones_like(alpha_hat)
        S_alpha = alpha_hat.sum(-1, keepdim=True)
        S_beta = beta.sum(-1, keepdim=True)

        ln_alpha = S_alpha.lgamma() - alpha_hat.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()

        dg_alpha = torch.digamma(alpha_hat)
        dg_S_alpha = torch.digamma(S_alpha)

        L_kl = (
            ln_alpha
            + ln_beta
            + torch.sum((alpha_hat - beta) * (dg_alpha - dg_S_alpha), -1, keepdim=True)
        )

        return (L_sos + self.v_kl * L_kl).mean(-1)


class DirichletClassificationLoss(DirichletLossBase):
    alias = "classification-dirichlet"

    def calc(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        num_tasks = targets.shape[1]
        num_classes = 2
        preds = preds.reshape(len(preds), num_tasks, num_classes)

        y_one_hot = torch.eye(num_classes, device=preds.device)[targets.long()]

        return super().calc(preds, y_one_hot)


class DirichletMulticlassLoss(DirichletLossBase):
    alias = "multiclass-dirichlet"

    def calc(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        y_one_hot = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        return super().calc(preds, y_one_hot, mask)


class SpectralLoss(LossFunction):
    def __init__(self, threshold: Optional[float] = None, **kwargs):
        self.threshold = threshold


class SIDSpectralLoss(SpectralLoss):
    alias = "spectral-sid"

    def calc(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
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

    def calc(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return torch.abs(targets.cumsum(1) - preds_norm.cumsum(1))


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
