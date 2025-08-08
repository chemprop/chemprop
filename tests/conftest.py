import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

_DATA_DIR = Path(__file__).parent / "data"
_DF = pd.read_csv(_DATA_DIR / "smis.csv")
_DF["mol"] = _DF["smiles"].map(Chem.MolFromSmiles)
_DF["smi"] = _DF["mol"].map(Chem.MolToSmiles)
# Ensure atom numbering is consistent between mol and smi for `test_same_featurization`
_DF["mol"] = _DF["smi"].map(Chem.MolFromSmiles)


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture
def smis():
    return _DF.smi


@pytest.fixture
def mols():
    return _DF.mol


@pytest.fixture
def targets(smis):
    return np.random.rand(len(smis), 1)


# @pytest.fixture
# def mol_data(mols, targets):
#     return [MoleculeDatapoint(mol, y) for mol, y in zip(mols, targets)]


# @pytest.fixture
# def rxn_data(rxns, targets):
#     return [ReactionDatapoint(mol, y) for mol, y in zip(mols, targets)]


@pytest.fixture(params=_DF.smi.sample(5))
def smi(request):
    return request.param


@pytest.fixture(params=_DF.mol.sample(5))
def mol(request):
    return request.param


@pytest.fixture
def mol_regression_data(data_dir):
    df = pd.read_csv(data_dir / "regression/mol/mol.csv")
    smis = df["smiles"].to_list()
    Y = df["lipo"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def mol_atom_bond_regression_data(data_dir):
    df = pd.read_csv(data_dir / "mol_atom_bond/regression.csv")
    columns = ["smiles", "mol_y1", "mol_y2", "atom_y1", "atom_y2", "bond_y1", "bond_y2"]
    smis = df.loc[:, columns[0]].values
    mol_ys = df.loc[:, columns[1:2]].values
    atoms_ys = df.loc[:, columns[3:4]].values
    bonds_ys = df.loc[:, columns[5:6]].values
    atoms_ys = [
        np.array([ast.literal_eval(atom_y) for atom_y in atom_ys], dtype=float).T
        for atom_ys in atoms_ys
    ]
    bonds_ys = [
        np.array([ast.literal_eval(bond_y) for bond_y in bond_ys], dtype=float).T
        for bond_ys in bonds_ys
    ]
    return smis, mol_ys, atoms_ys, bonds_ys


@pytest.fixture
def rxn_regression_data(data_dir):
    df = pd.read_csv(data_dir / "regression/rxn/rxn.csv")
    smis = df["smiles"].to_list()
    Y = df["ea"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def mol_mol_regression_data(data_dir):
    df = pd.read_csv(data_dir / "regression/mol+mol/mol+mol.csv")
    smis1 = df["smiles"].to_list()
    smis2 = df["solvent"].to_list()
    Y = df["peakwavs_max"].to_numpy().reshape(-1, 1)

    return smis1, smis2, Y


@pytest.fixture
def rxn_mol_regression_data(data_dir):
    df = pd.read_csv(data_dir / "regression/rxn+mol/rxn+mol.csv")
    rxns = df["rxn_smiles"].to_list()
    smis = df["solvent_smiles"].to_list()
    Y = df["target"].to_numpy().reshape(-1, 1)

    return rxns, smis, Y


@pytest.fixture
def mol_classification_data(data_dir):
    df = pd.read_csv(data_dir / "classification" / "mol.csv")
    smis = df["smiles"].to_list()
    Y = df["NR-AhR"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def mol_classification_data_multiclass(data_dir):
    df = pd.read_csv(data_dir / "classification" / "mol_multiclass.csv")
    smis = df["smiles"].to_list()
    activities = df["activity"].unique()
    Y = (
        df["activity"]
        .map({activity: i for i, activity in enumerate(activities)})
        .to_numpy()
        .reshape(-1, 1)
    )

    return smis, Y
