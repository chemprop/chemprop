"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    MulticomponentDataset,
    ReactionDatapoint,
    ReactionDataset,
    collate_multicomponent,
)
from chemprop.featurizers.molgraph import CondensedGraphOfReactionFeaturizer

N_COMPONENTS = 2
SHAPE = CondensedGraphOfReactionFeaturizer().shape
pytestmark = [
    pytest.mark.parametrize(
        "mcmpnn",
        [
            ([nn.BondMessagePassing(*SHAPE), nn.BondMessagePassing()], N_COMPONENTS, False),
            ([nn.AtomMessagePassing(*SHAPE), nn.AtomMessagePassing()], N_COMPONENTS, False),
        ],
        indirect=True,
    ),
    pytest.mark.integration,
]


@pytest.fixture
def datas(rxn_mol_regression_data):
    rxns, smis, Y = rxn_mol_regression_data

    return [
        [ReactionDatapoint.from_smi(smi, y) for smi, y in zip(rxns, Y)],
        [MoleculeDatapoint.from_smi(smi) for smi in smis],
    ]


@pytest.fixture
def dataloader(datas):
    dsets = [ReactionDataset(datas[0]), MoleculeDataset(datas[1])]
    dset = MulticomponentDataset(dsets)
    dset.normalize_targets()

    return DataLoader(dset, 32, collate_fn=collate_multicomponent)


def test_quick(mcmpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mcmpnn, dataloader)


def test_overfit(mcmpnn, dataloader):
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
    trainer.fit(mcmpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, targets, *_ = batch
        preds = mcmpnn(bmg)
        errors.append(preds - targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.05
