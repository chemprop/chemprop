from io import StringIO
import re

from lightning import pytorch as pl
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from chemprop.nn.metrics import (
    MAE,
    MSE,
    RMSE,
    SID,
    BCELoss,
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryF1Score,
    BinaryMCCLoss,
    BinaryMCCMetric,
    BoundedMAE,
    BoundedMSE,
    BoundedRMSE,
    CrossEntropyLoss,
    DirichletLoss,
    EvidentialLoss,
    MulticlassMCCLoss,
    MulticlassMCCMetric,
    MVELoss,
    R2Score,
    Wasserstein,
)

reg_targets = torch.arange(-20, 20, dtype=torch.float32).view(-1, 2)
# fmt: off
b_class_targets = torch.tensor(
    [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 
     1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32
).view(-1, 2)
m_class_targets = torch.tensor(
    [0, 2, 1, 0, 2, 0, 2, 2, 1, 0, 1, 1, 0, 1, 2, 1, 0, 0, 1, 0,
     0, 0, 0, 2, 1, 2, 2, 1, 2, 2, 2, 0, 1, 1, 0, 0, 1, 1, 2, 0], dtype=torch.float32
).view(-1, 2)
raw_spectra = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                            0, 1, 2, 3, 4, 5, 4, 3, 2, 1,
                            4, 3, 2, 1, 0, 1, 2, 3, 4, 5,
                            9, 1, 8, 0, 5, 4, 3, 6, 8, 3,
                            2, 1, 6, 4, 7, 2, 6, 2, 5, 1,
                            5, 3, 4, 4, 4, 4, 5, 1, 2, 8,
                            9, 7, 6, 5, 4, 3, 2, 0, 0, 0,
                            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                            4, 3, 4, 3, 4, 3, 4, 3, 4, 3,
                            9, 0, 1, 5, 2, 6, 2, 7, 4, 7,
                            3, 8, 2, 9, 1, 0, 1, 0, 1, 0,
                            4, 4, 2, 3, 1, 3, 2, 3, 1, 3,
                            9, 1, 8, 0, 5, 4, 3, 6, 8, 3], dtype=torch.float32).view(-1, 10)
spectral_targets = raw_spectra / raw_spectra.sum(1, keepdim=True)

mockffn = torch.tensor([
 -3.7, -14.2,   3.4,   7.5,  11.7,  13.8,  10.2,  -0.7,   9.2,  -8.0,
 -5.3,  -2.7,  -5.3, -14.4,   1.3,   9.0,  -0.4, -10.9,  14.8,  16.4,
 10.9,   5.8, -18.9,   3.6,  18.3,  -2.7, -16.8,  -8.4,   9.7,  -7.2,
 17.1,  -9.6,  -3.3,  -1.0, -11.9, -19.6, -12.3, -13.9,  -1.1,  -6.0,
  1.1,  12.0,  -7.8,   0.2, -12.9,  13.8,   1.1,  -9.4,   4.3, -14.9,
 10.0,   9.2,  -1.3,  -4.4,  -7.0,  18.5, -17.5,  -0.3, -13.2,  -0.1,
 16.2, -14.6, -19.6,   5.5,   4.7,  -4.5,  -4.9,  13.8,  12.3,  -6.9,
-12.1, -18.6,  -9.5,   9.8,  -9.6,  -9.9,   8.7,   0.5,  11.2,  13.0,
 -1.2,   4.2, -15.9,  11.4,  14.6, -19.9,  14.7,  -3.0, -10.0,   9.5,
  9.0,  -6.8, -13.0, -18.0, -12.6,   8.5,  16.9, -17.8, -11.2,  14.5,
-11.8,  -5.1,   5.1,   8.5,  -4.2,  11.6,  14.5,  19.7, -17.1,  19.0,
 19.2,  17.7,  -4.9,   0.7, -16.5,   2.9,  11.3,  -5.5,  17.8,  14.6,
 -4.2,  -1.4,  -7.3,   8.4,  -8.0,   2.5,  17.5,  13.3,  -6.0,  -7.9,
  3.5,  -2.8,   2.8,  15.3,  15.2,  -9.3,  -1.0, -20.0, -19.6, -16.7,
-15.5, -10.3, -16.6,  17.9,  18.3,   4.2, -15.8,   5.8,  13.0,   7.9,
 19.7,   7.7,  16.5,   1.8, -16.6,  -4.3,   2.9,  18.4,   4.2,  13.1,
    ], dtype=torch.float32,
)
# fmt: on

reg_train_step = mockffn.clone()[:40].view(-1, 2)
reg_forward = reg_train_step.clone()
mve_train_step = torch.stack(
    (mockffn.clone()[:40].view(-1, 2), F.softplus(mockffn.clone()[40:80].view(-1, 2))), 2
)
mve_forward = mve_train_step.clone()
evi_train_step = torch.stack(
    (
        mockffn.clone()[:40].view(-1, 2),
        F.softplus(mockffn.clone()[40:80].view(-1, 2)),
        F.softplus(mockffn.clone()[80:120].view(-1, 2)) + 1,
        F.softplus(mockffn.clone()[120:160].view(-1, 2)),
    ),
    2,
)
evi_forward = evi_train_step.clone()

b_class_train_step = mockffn.clone()[:40].view(-1, 2)
b_class_forward = b_class_train_step.clone().sigmoid()
b_diri_train_step = F.softplus(mockffn.clone()[0:80].view(-1, 2, 2)) + 1
b_diri_forward = b_diri_train_step[..., 1] / b_diri_train_step.sum(-1)

m_class_train_step = mockffn.clone()[:120].view(20, 2, 3)
m_class_forward = m_class_train_step.clone().softmax(-1)
m_diri_train_step = F.softplus(mockffn.clone()[:120].view(20, 2, 3)) + 1
m_diri_forward = m_diri_train_step / m_diri_train_step.sum(-1, keepdim=True)
spectral_train_step = mockffn.clone()[:150].view(-1, 10).exp() / mockffn.clone()[:150].view(
    -1, 10
).exp().sum(1, keepdim=True)
spectral_forward = spectral_train_step.clone()

# fmt: off
mask = torch.tensor(
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
     1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool
).view(-1, 2)
spectral_mask = torch.tensor(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
     1, 0, 0, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
     1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
     1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
     1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
     1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
     1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
     1, 1, 1, 0, 0, 1, 1, 1, 1, 1], dtype=torch.bool
).view(-1, 10)
# fmt: on


class _MockDataset(Dataset):
    def __init__(self, train_step, forward, targets, mask):
        self.train_step = train_step
        self.forward = forward
        self.targets = targets
        # fmt: off
        self.mask = mask
        self.w = torch.linspace(0.1, 1, len(self.targets), dtype=torch.float32).view(-1, 1)
        self.lt_mask = torch.tensor(
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
             0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1], dtype=torch.bool
        ).view(-1, 2)
        self.gt_mask = torch.tensor(
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
             0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool
        ).view(-1, 2)
        # fmt: on

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            self.train_step[idx],
            self.forward[idx],
            self.targets[idx],
            self.mask[idx],
            self.w[idx],
            self.lt_mask[idx],
            self.gt_mask[idx],
        )


class _MockMPNN(pl.LightningModule):
    def __init__(self, criterion, metric):
        super().__init__()
        self.automatic_optimization = False
        self.ignore = torch.nn.Parameter(torch.tensor(0.0))
        self.criterion = criterion
        self.metrics = torch.nn.ModuleList([metric, self.criterion.clone()])

    def training_step(self, batch, batch_idx):
        train_step, _, targets, mask, w, lt_mask, gt_mask = batch
        loss = self.criterion(train_step, targets, mask, w, lt_mask, gt_mask)
        self.log("train_loss", self.criterion, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._evalute_batch(batch, "val")

        train_step, _, targets, mask, w, lt_mask, gt_mask = batch
        self.metrics[-1].update(train_step, targets, mask, w, lt_mask, gt_mask)
        self.log("val_loss", self.metrics[-1], prog_bar=True)

    def test_step(self, batch, batch_idx):
        self._evalute_batch(batch, "test")

    def _evalute_batch(self, batch, val_test):
        _, forward, targets, mask, w, lt_mask, gt_mask = batch
        if isinstance(self.metrics[-1], (MVELoss, EvidentialLoss)):
            forward = forward[..., 0]
        self.metrics[0].update(forward, targets, mask, w, lt_mask, gt_mask)
        self.log(f"{val_test}_metric", self.metrics[0], prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class _TestBar(TQDMProgressBar):
    def __init__(self, bar_as_text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_as_text = bar_as_text

    def init_train_tqdm(self) -> Tqdm:
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=self.bar_as_text,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )


# fmt: off
groups = [
    (MSE(), R2Score(), reg_train_step, reg_forward, reg_targets, mask),
    (MAE(), MSE(), reg_train_step, reg_forward, reg_targets, mask),
    (RMSE(), MAE(), reg_train_step, reg_forward, reg_targets, mask),
    (BoundedMSE(), RMSE(), reg_train_step, reg_forward, reg_targets, mask),
    (BoundedMAE(), BoundedMSE(), reg_train_step, reg_forward, reg_targets, mask),
    (BoundedRMSE(), BoundedMAE(), reg_train_step, reg_forward, reg_targets, mask),
    (MSE(), BoundedRMSE(), reg_train_step, reg_forward, reg_targets, mask),
    (MVELoss(), MSE(), mve_train_step, mve_forward, reg_targets, mask),
    (EvidentialLoss(), MSE(), evi_train_step, evi_forward, reg_targets, mask),
    (BCELoss(), BinaryMCCMetric(), b_class_train_step, b_class_forward, b_class_targets, mask),
    (BinaryMCCLoss(), BinaryAUROC(), b_class_train_step, b_class_forward, b_class_targets, mask),
    (BCELoss(), BinaryAUPRC(), b_class_train_step, b_class_forward, b_class_targets, mask),
    (BCELoss(), BinaryAccuracy(), b_class_train_step, b_class_forward, b_class_targets, mask),
    (DirichletLoss(), BinaryF1Score(), b_diri_train_step, b_diri_forward, b_class_targets, mask),
    (CrossEntropyLoss(), MulticlassMCCMetric(), m_class_train_step, m_class_forward, m_class_targets, mask),
    (MulticlassMCCLoss(), MulticlassMCCMetric(), m_class_train_step, m_class_forward, m_class_targets, mask),
    (DirichletLoss(), MulticlassMCCMetric(), m_diri_train_step, m_diri_forward, m_class_targets, mask),
    (SID(), Wasserstein(), spectral_train_step, spectral_forward, spectral_targets, spectral_mask),
    (Wasserstein(), SID(), spectral_train_step, spectral_forward, spectral_targets, spectral_mask),
]
# fmt: on


@pytest.mark.parametrize("loss_fn, metric_fn, train_step, forward, targets, mask", groups)
def test_metric_integeration(loss_fn, metric_fn, train_step, forward, targets, mask):
    model = _MockMPNN(loss_fn, metric_fn)

    dataset = _MockDataset(train_step, forward, targets, mask)
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=5, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=20, shuffle=False)

    bar_as_text = StringIO()
    trainer = pl.Trainer(max_epochs=2, log_every_n_steps=1, callbacks=[_TestBar(bar_as_text)])
    trainer.fit(model, train_loader, val_loader)

    x = bar_as_text.getvalue()
    train_losses = re.findall(r"train_loss_epoch=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", x)
    val_losses = re.findall(r"val_loss=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", x)
    val_metrics = re.findall(r"val_metric=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", x)

    test_results = trainer.test(model, test_loader)
    test_metric = test_results[0]["test_metric"]

    for train_loss in train_losses:
        for val_loss in val_losses:
            train_loss, val_loss = float(train_loss), float(val_loss)
            assert abs(train_loss - val_loss) <= 0.01 * max(abs(train_loss), abs(val_loss))

    for value in val_metrics:
        assert abs(float(value) - test_metric) <= 0.01 * max(abs(float(value)), abs(test_metric))
