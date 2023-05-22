from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.v2.utils import ReprMixin, ClassFactory

LossFunctionFactory = ClassFactory()


class LossFunction(ABC, ReprMixin):
    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, w_s: Tensor, w_t: Tensor, **kwargs
    ):
        """Calculate the mean loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a tensor of shape `b x (t * u)` (regression), `b x t` (binary classification), or
            `b x t x c` (multiclass classification) containing the predictions, where `b` is the
            batch size, `t` is the number of tasks to predict, `u` is the number of
            targets to predict for each task, and `c` is the number of classes. 
        targets : Tensor
            a float tensor of shape `b x t` containing the target values
        mask : Tensor
            a boolean tensor of shape `b x t` indicating whether the given prediction should be
            included in the loss calculation
        w_s : Tensor
            a tensor of shape `b` or `b x 1` containing the per-sample weight
        w_t : Tensor
            a tensor of shape `t` or `1 x t` containing the per-task weight
        **kwargs
            keyword arguments specific to the given loss function

        Returns
        -------
        Tensor
            a scalar containing the fully reduced loss
        """
        L = self.forward(preds, targets, mask=mask, **kwargs)
        L = L * w_s.view(-1, 1) * w_t.view(1, -1) * mask

        return L.sum() / mask.sum()

    @abstractmethod
    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Calculate a tensor of shape `b x t` containing the unreduced loss values."""


@LossFunctionFactory.register("mse")
class MSELoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


@LossFunctionFactory.register("bounded-mse")
class BoundedMSELoss(MSELoss):
    def forward(
        self, preds: Tensor, targets: Tensor, lt_mask: Tensor, gt_mask: Tensor, **kwargs
    ) -> Tensor:
        preds[preds < targets & lt_mask] = targets
        preds[preds > targets & gt_mask] = targets

        return super().forward(preds, targets)


@LossFunctionFactory.register("mve")
class MVELoss(LossFunction):
    """Calculate the loss using Eq. 9 from [1]_

    References
    ----------
    .. [1] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target probability
    distribution." Proceedings of 1994 IEEE International Conference on Neural Networks, 1994
    https://doi.org/10.1109/icnn.1994.374138
    """

    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        mean, var = torch.chunk(preds, 2, 1)

        L_sos = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2

        return L_sos + L_kl


@LossFunctionFactory.register("evidential")
class EvidentialLoss(LossFunction):
    """
    References
    ----------
    .. [1] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.; "Evidential
    Deep Learning for Guided Molecular Property Prediction and Discovery." ACS Cent. Sci. 2021, 7,
    8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    def __init__(self, v_kl: float = 0.2, eps: float = 1e-8):
        self.v_kl = v_kl
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        mean, v, alpha, beta = torch.chunk(preds, 4, 1)

        residuals = targets - mean
        twoBlambda = 2 * beta * (1 + v)

        L_nll = (
            0.5 * (torch.pi / v).log()
            - alpha * twoBlambda.log()
            + (alpha + 0.5) * torch.log(v * residuals**2 + twoBlambda)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        L_reg = (2 * v + alpha) * residuals.abs()

        return L_nll + self.v_kl * (L_reg - self.eps)

    def get_params(self) -> list[tuple[str, float]]:
        return [("v_kl", self.v_kl), ("eps", self.eps)]


@LossFunctionFactory.register("binary-xent")
class BCELoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


@LossFunctionFactory.register("multiclass-xent")
class CrossEntropyLoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()

        return F.cross_entropy(preds, targets, reduction="none")


class MCCLossBase(LossFunction):
    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor, w_t: Tensor, **kwargs
    ):
        if not (0 <= preds.min() and preds.max() <= 1):  # assume logits
            preds = preds.softmax(2)

        L = self.forward(preds, targets.long(), mask=mask, w_d=w_d, **kwargs)
        L = L * w_t

        return L.mean()


@LossFunctionFactory.register("binary-mcc")
class BinaryMCCLoss(MCCLossBase):
    """Calculate a soft Matthews correlation coefficient loss for binary classification

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Phi_coefficient
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor) -> Tensor:
        TP = (targets * preds * w_d * mask).sum(0, keepdim=True)
        FP = ((1 - targets) * preds * w_d * mask).sum(0, keepdim=True)
        TN = ((1 - targets) * (1 - preds) * w_d * mask).sum(0, keepdim=True)
        FN = (targets * (1 - preds) * w_d * mask).sum(0, keepdim=True)

        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).sqrt()

        return 1 - MCC


@LossFunctionFactory.register("multiclass-mcc")
class MulticlassMCCLoss(MCCLossBase):
    """Calculate a soft Matthews correlation coefficient loss for multiclass classification

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, w_d: Tensor) -> Tensor:
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
    """Uses the loss function from [1]_ based on the implementation at [2]_

    References
    ----------
    .. [1] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
    classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    .. [2] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function
    """

    def __init__(self, v_kl: float = 0.2):
        self.v_kl = v_kl

    def forward(self, preds: Tensor, targets: Tensor, **kwargs) -> Tensor:
        S = preds.sum(-1, keepdim=True)
        p = preds / S

        A = ((targets - p) ** 2).sum(-1, keepdim=True)
        B = ((p * (1 - p)) / (S + 1)).sum(-1, keepdim=True)

        L_mse = A + B

        alpha = targets + (1 - targets) * preds
        beta = torch.ones_like(alpha)
        S_alpha = alpha.sum(-1, keepdim=True)
        S_beta = beta.sum(-1, keepdim=True)

        ln_alpha = S_alpha.lgamma() - alpha.lgamma().sum(-1, keepdim=True)
        ln_beta = beta.lgamma().sum(-1, keepdim=True) - S_beta.lgamma()

        dg0 = torch.digamma(alpha)
        dg1 = torch.digamma(S_alpha)

        L_kl = ln_alpha + ln_beta + torch.sum((alpha - beta) * (dg0 - dg1), -1, keepdim=True)

        return (L_mse + self.v_kl * L_kl).mean(-1)

    def get_params(self) -> list[tuple[str, float]]:
        return [("v_kl", self.v_kl)]


@LossFunctionFactory.register("binary-dirichlet")
class BinaryDirichletLoss(DirichletLossBase):
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        num_tasks = targets.shape[1]
        num_classes = 2
        preds = preds.reshape(len(preds), num_tasks, num_classes)

        y_one_hot = torch.eye(num_classes, device=preds.device)[targets.long()]

        return super().forward(preds, y_one_hot)


@LossFunctionFactory.register("multiclass-dirichlet")
class MulticlassDirichletLoss(DirichletLossBase):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        y_one_hot = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        return super().forward(preds, y_one_hot, mask)


class SpectralLoss(LossFunction):
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def get_params(self) -> list[tuple[str, float]]:
        return [("threshold", self.threshold)]


@LossFunctionFactory.register("sid")
class SIDLoss(SpectralLoss):
    alias = "spectral-sid"

    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (preds_norm / targets).log() * preds_norm + (targets / preds_norm).log() * targets


@LossFunctionFactory.register("earthmovers")
class WassersteinLoss(SpectralLoss):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, **kwargs) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return (targets.cumsum(1) - preds_norm.cumsum(1)).abs()
