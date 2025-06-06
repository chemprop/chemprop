"""This integration test is designed to ensure that the chemprop model can _overfit_ the training
data. A small enough dataset should be memorizable by even a moderately sized model, so this test
should generally pass."""

from lightning import pytorch as pl
import pytest
from torch.utils.data import DataLoader

from chemprop import models, nn
from chemprop.data import MolAtomBondDatapoint, MolAtomBondDataset, collate_mol_atom_bond_batch


@pytest.fixture
def dataloader(mol_atom_bond_regression_data):
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


def test_mol_atom_bond(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing()
    agg = nn.NormAggregation()
    mol_predictor = nn.RegressionFFN()
    atom_predictor = nn.RegressionFFN()
    bond_predictor = nn.RegressionFFN(input_dim=600)
    model = models.MolAtomBondMPNN(
        message_passing=mp,
        agg=agg,
        mol_predictor=mol_predictor,
        atom_predictor=atom_predictor,
        bond_predictor=bond_predictor,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_mol_atom_none(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing(return_edge_embeddings=False)
    agg = nn.SumAggregation()
    mol_predictor = nn.RegressionFFN()
    atom_predictor = nn.RegressionFFN()
    model = models.MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=mol_predictor, atom_predictor=atom_predictor
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_mol_none_bond(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing()
    agg = nn.MeanAggregation()
    mol_predictor = nn.RegressionFFN()
    bond_predictor = nn.RegressionFFN(input_dim=600)
    model = models.MolAtomBondMPNN(
        message_passing=mp, agg=agg, mol_predictor=mol_predictor, bond_predictor=bond_predictor
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_none_atom_bond(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing()
    atom_predictor = nn.RegressionFFN()
    bond_predictor = nn.RegressionFFN(input_dim=600)
    model = models.MolAtomBondMPNN(
        message_passing=mp, atom_predictor=atom_predictor, bond_predictor=bond_predictor
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_mol_none_none(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing(return_edge_embeddings=False)
    agg = nn.NormAggregation()
    mol_predictor = nn.RegressionFFN()
    model = models.MolAtomBondMPNN(message_passing=mp, agg=agg, mol_predictor=mol_predictor)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_none_atom_none(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing(return_edge_embeddings=False)
    atom_predictor = nn.RegressionFFN()
    model = models.MolAtomBondMPNN(message_passing=mp, atom_predictor=atom_predictor)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_none_none_bond(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing(return_vertex_embeddings=False)
    bond_predictor = nn.RegressionFFN(input_dim=600)
    model = models.MolAtomBondMPNN(message_passing=mp, bond_predictor=bond_predictor)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
    )
    trainer.fit(model, dataloader, None)

    models.utils.save_model(tmp_path / "temp.pt", model)
    models.MolAtomBondMPNN.load_from_file(tmp_path / "temp.pt")


def test_none_none_none(dataloader, tmp_path):
    mp = nn.MABBondMessagePassing()
    with pytest.raises(ValueError):
        models.MolAtomBondMPNN(message_passing=mp)
