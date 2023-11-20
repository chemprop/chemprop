from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.v2.utils import ClassRegistry, ReprMixin

LossFunctionRegistry = ClassRegistry()


class LossFunction(ABC, ReprMixin):
    def __call__(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        w_s: Tensor,
        w_t: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ):
        """Calculate the mean loss function value given predicted and target values

        Parameters
        ----------
        preds : Tensor
            a tensor of shape `b x (t * s)` (regression), `b x t` (binary classification), or
            `b x t x c` (multiclass classification) containing the predictions, where `b` is the
            batch size, `t` is the number of tasks to predict, `s` is the number of
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
        lt_mask: Tensor
        gt_mask: Tensor

        Returns
        -------
        Tensor
            a scalar containing the fully reduced loss
        """
        L = self.forward(preds, targets, mask, w_s, w_t, lt_mask, gt_mask)
        L = L * w_s.view(-1, 1) * w_t.view(1, -1) * mask

        return L.sum() / mask.sum()

    @abstractmethod
    def forward(self, preds, targets, mask, w_s, w_t, lt_mask, gt_mask) -> Tensor:
        """Calculate a tensor of shape `b x t` containing the unreduced loss values."""


@LossFunctionRegistry.register("mse")
class MSELoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.mse_loss(preds, targets, reduction="none")


@LossFunctionRegistry.register("bounded-mse")
class BoundedMSELoss(MSELoss):
    def forward(
        self, preds: Tensor, targets: Tensor, mask, w_s, w_t, lt_mask: Tensor, gt_mask: Tensor
    ) -> Tensor:
        preds = torch.where((preds < targets) & lt_mask, targets, preds)
        preds = torch.where((preds > targets) & gt_mask, targets, preds)

        return super().forward(preds, targets)


@LossFunctionRegistry.register("mve")
class MVELoss(LossFunction):
    """Calculate the loss using Eq. 9 from [nix1994]_

    References
    ----------
    .. [nix1994] Nix, D. A.; Weigend, A. S. "Estimating the mean and variance of the target
        probability distribution." Proceedings of 1994 IEEE International Conference on Neural
        Networks, 1994 https://doi.org/10.1109/icnn.1994.374138
    """

    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        mean, var = torch.chunk(preds, 2, 1)

        L_sos = (mean - targets) ** 2 / (2 * var)
        L_kl = (2 * torch.pi * var).log() / 2

        return L_sos + L_kl


@LossFunctionRegistry.register("evidential")
class EvidentialLoss(LossFunction):
    """Caculate the loss using Eq. **TODO** from [soleimany2021]_

    References
    ----------
    .. [soleimany2021] Soleimany, A.P.; Amini, A.; Goldman, S.; Rus, D.; Bhatia, S.N.; Coley, C.W.;
        "Evidential Deep Learning for Guided Molecular Property Prediction and Discovery." ACS
        Cent. Sci. 2021, 7, 8, 1356-1367. https://doi.org/10.1021/acscentsci.1c00546
    """

    def __init__(self, v_kl: float = 0.2, eps: float = 1e-8):
        self.v_kl = v_kl
        self.eps = eps

    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
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


@LossFunctionRegistry.register("bce")
class BCELoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return F.binary_cross_entropy_with_logits(preds, targets, reduction="none")


@LossFunctionRegistry.register("ce")
class CrossEntropyLoss(LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        preds = preds.transpose(1, 2)
        targets = targets.long()

        return F.cross_entropy(preds, targets, reduction="none")


class MccMixin:
    """Calculate a soft Matthews correlation coefficient ([mccWiki]_) loss for multiclass
    classification based on the implementataion of [mccSklearn]_

    References
    ----------
    .. [mccWiki] https://en.wikipedia.org/wiki/Phi_coefficient#Multiclass_case
    .. [mccSklearn] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html
    """

    def __call__(
        self, preds: Tensor, targets: Tensor, mask: Tensor, w_s: Tensor, w_t: Tensor, *args
    ):
        if not (0 <= preds.min() and preds.max() <= 1):  # assume logits
            preds = preds.softmax(2)

        L = self.forward(preds, targets.long(), mask, w_s, *args)
        L = L * w_t

        return L.mean()


@LossFunctionRegistry.register("binary-mcc")
class BinaryMCCLoss(LossFunction, MccMixin):
    def forward(self, preds, targets, mask, w_s, *args) -> Tensor:
        TP = (targets * preds * w_s * mask).sum(0, keepdim=True)
        FP = ((1 - targets) * preds * w_s * mask).sum(0, keepdim=True)
        TN = ((1 - targets) * (1 - preds) * w_s * mask).sum(0, keepdim=True)
        FN = (targets * (1 - preds) * w_s * mask).sum(0, keepdim=True)

        MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)).sqrt()

        return 1 - MCC


@LossFunctionRegistry.register("multiclass-mcc")
class MulticlassMCCLoss(LossFunction, MccMixin):
    def forward(self, preds, targets, mask, w_s, *args) -> Tensor:
        device = preds.device

        C = preds.shape[2]
        bin_targets = torch.eye(C, device=device)[targets]
        bin_preds = torch.eye(C, device=device)[preds.argmax(-1)]
        masked_data_weights = w_s.unsqueeze(2) * mask.unsqueeze(2)

        p = (bin_preds * masked_data_weights).sum(0)
        t = (bin_targets * masked_data_weights).sum(0)
        c = (bin_preds * bin_targets * masked_data_weights).sum()
        s = (preds * masked_data_weights).sum()
        s2 = s.square()

        # the `einsum` calls amount to calculating the batched dot product
        cov_ytyp = c * s - torch.einsum("ij,ij->i", p, t).sum()
        cov_ypyp = s2 - torch.einsum("ij,ij->i", p, p).sum()
        cov_ytyt = s2 - torch.einsum("ij,ij->i", t, t).sum()

        x = cov_ypyp * cov_ytyt
        MCC = torch.tensor(0.0, device=device) if x == 0 else cov_ytyp / x.sqrt()

        return 1 - MCC


class DirichletMixin:
    """Uses the loss function from [sensoy2018]_ based on the implementation at [sensoyGithub]_

    References
    ----------
    .. [sensoy2018] Sensoy, M.; Kaplan, L.; Kandemir, M. "Evidential deep learning to quantify
        classification uncertainty." NeurIPS, 2018, 31. https://doi.org/10.48550/arXiv.1806.01768
    .. [sensoyGithub] https://muratsensoy.github.io/uncertainty.html#Define-the-loss-function
    """

    def __init__(self, v_kl: float = 0.2):
        self.v_kl = v_kl

    def forward(self, preds, targets, *args) -> Tensor:
        S = preds.sum(-1, keepdim=True)
        p = preds / S

        A = (targets - p).square().sum(-1, keepdim=True)
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


@LossFunctionRegistry.register("binary-dirichlet")
class BinaryDirichletLoss(DirichletMixin, LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        N_CLASSES = 2
        n_tasks = targets.shape[1]
        preds = preds.reshape(len(preds), n_tasks, N_CLASSES)
        y_one_hot = torch.eye(N_CLASSES, device=preds.device)[targets.long()]

        return super().forward(preds, y_one_hot, *args)


@LossFunctionRegistry.register("multiclass-dirichlet")
class MulticlassDirichletLoss(DirichletMixin, LossFunction):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        y_one_hot = torch.eye(preds.shape[2], device=preds.device)[targets.long()]

        return super().forward(preds, y_one_hot, mask)


@dataclass
class _ThresholdMixin:
    threshold: float | None = None

    def get_params(self) -> list[tuple[str, float]]:
        return [("threshold", self.threshold)]


@LossFunctionRegistry.register("sid")
class SIDLoss(LossFunction, _ThresholdMixin):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        targets = targets.masked_fill(~mask, 1)
        preds_norm = preds_norm.masked_fill(~mask, 1)

        return (preds_norm / targets).log() * preds_norm + (targets / preds_norm).log() * targets


@LossFunctionRegistry.register(["earthmovers", "wasserstein"])
class WassersteinLoss(LossFunction, _ThresholdMixin):
    def forward(self, preds: Tensor, targets: Tensor, mask: Tensor, *args) -> Tensor:
        if self.threshold is not None:
            preds = preds.clamp(min=self.threshold)

        preds_norm = preds / (preds * mask).sum(1, keepdim=True)

        return (targets.cumsum(1) - preds_norm.cumsum(1)).abs()
