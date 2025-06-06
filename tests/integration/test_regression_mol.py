"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, collate_batch


@pytest.fixture
def data(mol_regression_data):
    smis, Y = mol_regression_data

    return [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = MoleculeDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_batch)


@pytest.mark.parametrize(
    "mpnn",
    [
        (nn.BondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (nn.BondMessagePassing(), nn.NormAggregation()),
        (nn.BondMessagePassing(), nn.MeanAggregation(), torch.nn.Softplus()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_quick(mpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, dataloader, None)


@pytest.mark.parametrize(
    "mpnn",
    [
        (nn.BondMessagePassing(), nn.MeanAggregation()),
        (nn.AtomMessagePassing(), nn.SumAggregation()),
        (nn.BondMessagePassing(), nn.NormAggregation()),
        (nn.BondMessagePassing(), nn.MeanAggregation(), torch.nn.Softplus()),
    ],
    indirect=True,
)
@pytest.mark.integration
def test_overfit(mpnn, dataloader):
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
    trainer.fit(mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = mpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05


@pytest.mark.parametrize(
    "regression_mpnn_mve", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_mve_quick(regression_mpnn_mve, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_mve, dataloader, None)


@pytest.mark.parametrize(
    "regression_mpnn_evidential", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_evidential_quick(regression_mpnn_evidential, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_evidential, dataloader, None)


@pytest.mark.parametrize(
    "regression_mpnn_quantile", [nn.BondMessagePassing(), nn.AtomMessagePassing()], indirect=True
)
@pytest.mark.integration
def test_quantile_quick(regression_mpnn_quantile, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(regression_mpnn_quantile, dataloader, None)
