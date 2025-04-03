"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import (
    MolAtomBondDataset,
    MolAtomBondDatapoint,
    collate_mol_atom_bond_batch,
)

pytestmark = [
    pytest.mark.parametrize(
        "mol_atom_bond_mpnn",
        [
            (nn.MixedAtomMessagePassing(), nn.MeanAggregation()),
            (nn.MixedBondMessagePassing(), nn.MeanAggregation()),
        ],
        indirect=True,
    ),
    pytest.mark.integration,
]


@pytest.fixture
def dataloader(mixed_regression_data):
    smis, mol_Y, atom_Y, bond_Y = mixed_regression_data
    all_data = [MolAtomBondDatapoint.from_smi(smi, y, atom_y=atom_y, bond_y=bond_y, keep_h=True) for smi, y, atom_y, bond_y in zip(smis, mol_Y, atom_Y, bond_Y)]
    dset = MolAtomBondDataset(all_data)
    return DataLoader(dset, 32, collate_fn=collate_mol_atom_bond_batch)


def test_quick(mol_atom_bond_mpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(mol_atom_bond_mpnn, dataloader, None)


def test_overfit(mol_atom_bond_mpnn, dataloader):
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=130,
        overfit_batches=1.00,
    )
    trainer.fit(mol_atom_bond_mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, _, targets, *_ = batch
        preds = mol_atom_bond_mpnn(bmg)
        print(targets)
        errors.append(preds[0] - targets[0])
        errors.append(preds[1] - targets[1])
        preds[2] = (preds[2][::2] + preds[2][1::2]) / 2
        print(preds)
        errors.append(preds[2] - targets[2])

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.1
