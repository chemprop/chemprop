from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

_DATA_DIR = Path(__file__).parent / "data"
_DF = pd.read_csv(_DATA_DIR / "smis.csv")
_DF["mol"] = _DF["smiles"].map(Chem.MolFromSmiles)
_DF["smi"] = _DF["mol"].map(Chem.MolToSmiles)


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
