"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import ReactionDatapoint, ReactionDataset, collate_batch
from chemprop.featurizers.molgraph import CondensedGraphOfReactionFeaturizer

SHAPE = CondensedGraphOfReactionFeaturizer().shape
pytestmark = pytest.mark.parametrize(
    "mpnn",
    [
        (nn.BondMessagePassing(*SHAPE), nn.MeanAggregation()),
        (nn.AtomMessagePassing(*SHAPE), nn.SumAggregation()),
        (nn.BondMessagePassing(*SHAPE), nn.NormAggregation()),
    ],
    indirect=True,
)


@pytest.fixture
def data(rxn_regression_data):
    smis, Y = rxn_regression_data

    return [ReactionDatapoint.from_smi(smi, y) for smi, y in zip(smis, Y)]


@pytest.fixture
def dataloader(data):
    dset = ReactionDataset(data)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_batch)


def test_quick(dataloader, mpnn):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mpnn, dataloader)


def test_overfit(dataloader, mpnn):
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

    with torch.inference_mode():
        errors = []
        for batch in dataloader:
            bmg, _, _, targets, *_ = batch
            preds = mpnn(bmg)
            errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.01
