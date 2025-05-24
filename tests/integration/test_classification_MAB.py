import ast

from lightning import pytorch as pl
import numpy as np
import pandas as pd
import pytest

from chemprop import data, models, nn

columns = ["smiles", "mol_y1", "mol_y2", "atom_y1", "atom_y2", "bond_y1", "bond_y2", "weight"]


@pytest.fixture
def mab_data_dir(data_dir):
    return data_dir / "mol_atom_bond"


def make_dataloader(path):
    pl.seed_everything(0)
    df_input = pd.read_csv(path)
    smis = df_input.loc[:, columns[0]].values
    mol_ys = df_input.loc[:, columns[1:3]].values
    atoms_ys = df_input.loc[:, columns[3:5]].values
    bonds_ys = df_input.loc[:, columns[5:7]].values

    atoms_ys = [
        np.array([ast.literal_eval(atom_y) for atom_y in atom_ys], dtype=float).T
        for atom_ys in atoms_ys
    ]
    bonds_ys = [
        np.array([ast.literal_eval(bond_y) for bond_y in bond_ys], dtype=float).T
        for bond_ys in bonds_ys
    ]

    datapoints = [
        data.MolAtomBondDatapoint.from_smi(
            smi,
            keep_h=True,
            add_h=False,
            reorder_atoms=True,
            y=mol_ys[i],
            atom_y=atoms_ys[i],
            bond_y=bonds_ys[i],
        )
        for i, smi in enumerate(smis)
    ]
    dataset = data.MolAtomBondDataset(datapoints)
    return data.build_dataloader(dataset, shuffle=False, batch_size=4)


def test_classification_overfit(mab_data_dir):
    pl.seed_everything(0)
    dataloader = make_dataloader(mab_data_dir / "classification.csv")
    mp = nn.MABBondMessagePassing()
    agg = nn.SumAggregation()
    mol_predictor = nn.BinaryClassificationFFN(n_tasks=2)
    atom_predictor = nn.BinaryClassificationFFN(n_tasks=2)
    bond_predictor = nn.BinaryClassificationFFN(input_dim=600, n_tasks=2)
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
        max_epochs=20,
        overfit_batches=1.0,
        deterministic=True,
    )

    trainer.fit(model, dataloader)
    results = trainer.test(model, dataloader)
    auroc = sum(results[0].values())
    assert auroc > 2.97


def test_multiclass_overfit(mab_data_dir):
    pl.seed_everything(0)
    dataloader = make_dataloader(mab_data_dir / "multiclass.csv")
    mp = nn.MABBondMessagePassing()
    agg = nn.SumAggregation()
    mol_predictor = nn.MulticlassClassificationFFN(n_tasks=2, n_classes=3)
    atom_predictor = nn.MulticlassClassificationFFN(n_tasks=2, n_classes=3)
    bond_predictor = nn.MulticlassClassificationFFN(input_dim=600, n_tasks=2, n_classes=3)
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
        enable_progress_bar=True,
        max_epochs=40,
        overfit_batches=1.0,
        deterministic=True,
    )

    trainer.fit(model, dataloader)
    results = trainer.test(model, dataloader)
    mcc = sum(results[0].values())
    assert mcc > 2.97
