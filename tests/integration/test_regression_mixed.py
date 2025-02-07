"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
import torch
from torch.utils.data import DataLoader

from chemprop import nn
from chemprop.data import (
    AtomDataset,
    BondDataset,
    MockDataset,
    MolAtomBondDataset,
    MoleculeDatapoint,
    MoleculeDataset,
    mixed_collate_batch,
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
def data(mixed_regression_data):
    smis, mol_Y, atom_Y, bond_Y, atom_slices, bond_slices = mixed_regression_data
    all_data = []
    all_data.append(
        [MoleculeDatapoint.from_smi(smi, y, keep_h=True) for smi, y in zip(smis, mol_Y)]
    )

    atom_datapoints = []
    for i in range(len(smis)):
        smi = smis[i]
        y = atom_Y[atom_slices[i]:atom_slices[i+1]]
        atom_datapoints.append(MoleculeDatapoint.from_smi(smi, y, keep_h=True))
    all_data.append(atom_datapoints)

    bond_datapoints = []
    for i in range(len(smis)):
        smi = smis[i]
        y = bond_Y[bond_slices[i]:bond_slices[i+1]]
        bond_datapoints.append(MoleculeDatapoint.from_smi(smi, y, keep_h=True))
    all_data.append(bond_datapoints)
    return all_data


@pytest.fixture
def dataloader(data):
    dsets = []
    if data[0]:
        dset = MoleculeDataset(data[0])
        dset.normalize_targets()
        dsets.append(dset)
    else:
        dsets.append(MockDataset())

    if data[1]:
        dset = AtomDataset(data[1])
        dset.normalize_targets()
        dsets.append(dset)
    else:
        dsets.append(MockDataset())

    if data[2]:
        dset = BondDataset(data[2])
        dset.normalize_targets()
        dsets.append(dset)
    else:
        dsets.append(MockDataset())

    dset = MolAtomBondDataset(dsets[0], dsets[1], dsets[2])
    return DataLoader(dset, 32, collate_fn=mixed_collate_batch)


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
        bmg, _, _, _, mol_targets, *_ = batch[0]
        _, _, _, _, atom_targets, *_ = batch[1]
        _, _, _, _, bond_targets, *_ = batch[2]
        preds = mol_atom_bond_mpnn(bmg)
        errors.append(preds[0] - mol_targets)
        errors.append(preds[1] - atom_targets)
        preds[2] = (preds[2][::2] + preds[2][1::2]) / 2
        errors.append(preds[2] - bond_targets)

    errors = torch.cat(errors)
    mse = errors.square().mean().item()

    assert mse <= 0.1
