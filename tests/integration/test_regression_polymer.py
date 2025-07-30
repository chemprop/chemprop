from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import PolymerDatapoint, PolymerDataset, collate_polymer_batch


@pytest.fixture
def data(polymer_regression_data):
    smis, Y = polymer_regression_data

    return [PolymerDatapoint.from_smi(smi, y=y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = PolymerDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_polymer_batch)


@pytest.mark.parametrize(
    "wmpnn",
    [
        (nn.WeightedBondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (nn.WeightedBondMessagePassing(), nn.NormAggregation()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_quick(wmpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(wmpnn, dataloader, None)


@pytest.mark.parametrize(
    "wmpnn",
    [
        (nn.WeightedBondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.MeanAggregation()),
        (nn.WeightedBondMessagePassing(), nn.NormAggregation()),
        (nn.WeightedBondMessagePassing(), nn.SumAggregation()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_overfit(wmpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=100,
        overfit_batches=1.00,
    )
    trainer.fit(wmpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = wmpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05


@pytest.mark.parametrize(
    "regression_wmpnn_mve", [nn.WeightedBondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_mve_quick(regression_wmpnn_mve, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_wmpnn_mve, dataloader, None)


@pytest.mark.parametrize(
    "regression_wmpnn_evidential", [nn.WeightedBondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_evidential_quick(regression_wmpnn_evidential, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_wmpnn_evidential, dataloader, None)
