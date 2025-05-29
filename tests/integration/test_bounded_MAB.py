import ast
from pathlib import Path

from lightning import pytorch as pl
import numpy as np
import pandas as pd
import pytest

from chemprop import data, featurizers, models, nn


@pytest.fixture
def dataloader():
    pl.seed_everything(0)
    data_dir = Path(__file__).parent.parent / "data" / "mol_atom_bond"
    df_input = pd.read_csv(data_dir / "bounded.csv")
    columns = ["smiles", "mol_y1", "mol_y2", "atom_y1", "atom_y2", "bond_y1", "bond_y2", "weight"]
    smis = df_input.loc[:, columns[0]].values
    mol_ys = df_input.loc[:, columns[1:3]]
    atoms_ys = df_input.loc[:, columns[3:5]]
    bonds_ys = df_input.loc[:, columns[5:7]]

    mol_ys = mol_ys.astype(str)
    lt_mask = mol_ys.map(lambda x: "<" in x).to_numpy()
    gt_mask = mol_ys.map(lambda x: ">" in x).to_numpy()
    mol_ys = mol_ys.map(lambda x: x.strip("<").strip(">")).to_numpy(np.single)

    atoms_ys = atoms_ys.map(ast.literal_eval)
    atom_lt_masks = atoms_ys.map(lambda L: ["<" in v if v else False for v in L])
    atom_gt_masks = atoms_ys.map(lambda L: [">" in v if v else False for v in L])

    atom_lt_masks = atom_lt_masks.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()
    atom_gt_masks = atom_gt_masks.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()
    atoms_ys = atoms_ys.map(
        lambda L: np.array([v.strip("<").strip(">") if v else "nan" for v in L], dtype=np.single)
    )
    atoms_ys = atoms_ys.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()

    bonds_ys = bonds_ys.map(ast.literal_eval)
    bond_lt_masks = bonds_ys.map(lambda L: ["<" in v if v else False for v in L])
    bond_gt_masks = bonds_ys.map(lambda L: [">" in v if v else False for v in L])

    bond_lt_masks = bond_lt_masks.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()
    bond_gt_masks = bond_gt_masks.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()

    bond_lt_masks = [bond_lt_mask.astype(bool) for bond_lt_mask in bond_lt_masks]
    bond_gt_masks = [bond_gt_mask.astype(bool) for bond_gt_mask in bond_gt_masks]

    bonds_ys = bonds_ys.map(
        lambda L: np.array([v.strip("<").strip(">") if v else "nan" for v in L], dtype=np.single)
    )
    bonds_ys = bonds_ys.apply(lambda row: np.vstack(row.values).T, axis=1).tolist()

    datapoints = [
        data.MolAtomBondDatapoint.from_smi(
            smi,
            keep_h=True,
            add_h=False,
            reorder_atoms=True,
            y=mol_ys[i],
            atom_y=atoms_ys[i],
            bond_y=bonds_ys[i],
            lt_mask=lt_mask[i],
            gt_mask=gt_mask[i],
            atom_lt_mask=atom_lt_masks[i],
            atom_gt_mask=atom_gt_masks[i],
            bond_lt_mask=bond_lt_masks[i],
            bond_gt_mask=bond_gt_masks[i],
        )
        for i, smi in enumerate(smis)
    ]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    dataset = data.MolAtomBondDataset(datapoints, featurizer=featurizer)
    return data.build_dataloader(dataset, shuffle=False)


@pytest.fixture
def mol_atom_bond_mpnn():
    pl.seed_everything(0)
    mp = nn.MABAtomMessagePassing()
    metrics = [nn.BoundedMSE()]
    agg = nn.SumAggregation()
    mol_predictor = nn.RegressionFFN(n_tasks=2, criterion=nn.BoundedMSE())
    atom_predictor = nn.RegressionFFN(n_tasks=2, criterion=nn.BoundedMSE())
    bond_predictor = nn.RegressionFFN(
        input_dim=(mp.output_dims[1] * 2), n_tasks=2, criterion=nn.BoundedMSE()
    )
    model = models.MolAtomBondMPNN(
        message_passing=mp,
        agg=agg,
        mol_predictor=mol_predictor,
        atom_predictor=atom_predictor,
        bond_predictor=bond_predictor,
        metrics=metrics,
    )
    return model


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
    pl.seed_everything(0)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
        max_epochs=400,
        overfit_batches=1.00,
        deterministic=True,
    )
    trainer.fit(mol_atom_bond_mpnn, dataloader)
    results = trainer.test(mol_atom_bond_mpnn, dataloader)
    mse = sum(results[0].values())
    print(mse)

    assert mse <= 0.07
