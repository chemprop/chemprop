"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import models, nn
from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset, collate_mol_atom_bond_batch


@pytest.fixture
def dataloader(mol_atom_bond_regression_data):
    pl.seed_everything(0)
    smis, mols_ys, atoms_ys, bonds_ys = mol_atom_bond_regression_data
    all_data = [
        MolAtomBondDatapoint.from_smi(
            smi,
            keep_h=True,
            add_h=False,
            reorder_atoms=True,
            y=mol_ys,
            atom_y=atom_ys,
            bond_y=bond_ys,
        )
        for smi, mol_ys, atom_ys, bond_ys in zip(smis, mols_ys, atoms_ys, bonds_ys)
    ]
    dset = MolAtomBondDataset(all_data)
    return DataLoader(dset, 32, collate_fn=collate_mol_atom_bond_batch)


@pytest.mark.parametrize(
    "mol_atom_bond_mpnn",
    [
        (nn.MABAtomMessagePassing(), nn.MeanAggregation()),
        (nn.MABBondMessagePassing(), nn.MeanAggregation()),
    ],
    indirect=True,
)
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


pl.seed_everything(0)


@pytest.mark.parametrize(
    "mol_atom_bond_mpnn",
    [
        (nn.MABAtomMessagePassing(), nn.MeanAggregation()),
        (nn.MABBondMessagePassing(), nn.MeanAggregation()),
    ],
    indirect=True,
)
def test_overfit(mol_atom_bond_mpnn, dataloader):
    pl.seed_everything(0)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=150,
        overfit_batches=1.00,
        deterministic=True,
    )
    trainer.fit(mol_atom_bond_mpnn, dataloader)

    errors = []
    for batch in dataloader:
        bmg, _, _, _, targets, *_ = batch
        preds = mol_atom_bond_mpnn(bmg)
        errors.append(preds[0] - targets[0])
        errors.append(preds[1] - targets[1])
        errors.append(preds[2] - targets[2])

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.01  # note this is in the scaled target space


def test_mve_quick(dataloader):
    mp = nn.MABBondMessagePassing()
    agg = nn.SumAggregation()
    mol_ffn = nn.MveFFN()
    atom_ffn = nn.MveFFN()
    bond_ffn = nn.MveFFN(input_dim=600)

    MAB_mve = models.MolAtomBondMPNN(mp, agg, mol_ffn, atom_ffn, bond_ffn, batch_norm=True)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(MAB_mve, dataloader, None)
