from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path("tests/data")


@pytest.fixture
def mol_regression_data():
    df = pd.read_csv(DATA_PATH / "regression/mol.csv")
    smis = df["smiles"].to_list()
    Y = df["lipo"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def rxn_regression_data():
    df = pd.read_csv(DATA_PATH / "regression/rxn.csv")
    smis = df["smiles"].to_list()
    Y = df["ea"].to_numpy().reshape(-1, 1)

    return smis, Y


@pytest.fixture
def mol_mol_regression_data():
    df = pd.read_csv(DATA_PATH / "regression/mol+mol.csv")
    smis1 = df["smiles"].to_list()
    smis2 = df["solvent"].to_list()
    Y = df["peakwavs_max"].to_numpy().reshape(-1, 1)

    return smis1, smis2, Y


@pytest.fixture
def rxn_mol_regression_data():
    df = pd.read_csv(DATA_PATH / "regression/rxn+mol.csv")
    rxns = df["rxn_smiles"].to_list()
    smis = df["solven_smiles"].to_list()
    Y = df["target"].to_numpy().reshape(-1, 1)

    return rxns, smis, Y
