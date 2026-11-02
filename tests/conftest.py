import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from torch.utils._pytree import GetAttrKey, register_pytree_node

from chemprop.data import BatchMolGraph

_DATA_DIR = Path(__file__).parent / "data"
_DF = pd.read_csv(_DATA_DIR / "smis.csv")
_DF["mol"] = _DF["smiles"].map(Chem.MolFromSmiles)
_DF["smi"] = _DF["mol"].map(Chem.MolToSmiles)
# Ensure atom numbering is consistent between mol and smi for `test_same_featurization`
_DF["mol"] = _DF["smi"].map(Chem.MolFromSmiles)


@pytest.fixture(scope="session")
def batch_mol_graph_pytree():
    def flatten(bmg: BatchMolGraph):
        return [bmg.V, bmg.E, bmg.edge_index, bmg.rev_edge_index, bmg.batch], len(bmg)

    def unflatten(children, size):
        V, E, edge_index, rev_edge_index, batch = children
        bmg = object.__new__(BatchMolGraph)
        bmg.V = V
        bmg.E = E
        bmg.edge_index = edge_index
        bmg.rev_edge_index = rev_edge_index
        bmg.batch = batch
        bmg._BatchMolGraph__size = size
        return bmg

    def flatten_with_keys(bmg: BatchMolGraph):
        children, context = flatten(bmg)
        keys = [
            GetAttrKey("V"),
            GetAttrKey("E"),
            GetAttrKey("edge_index"),
            GetAttrKey("rev_edge_index"),
            GetAttrKey("batch"),
        ]
        return list(zip(keys, children)), context

    register_pytree_node(
        BatchMolGraph,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name="chemprop.data.collate.BatchMolGraph",
    )


# model files saved before Chemprop v2.4 contain arbitrary objects, so they must be loaded with
# `weights_only=False`, which can execute arbitrary code. Tests that load them must be marked with
# `unsafe_load` and only run if `--allow-unsafe-load` is passed.
def pytest_addoption(parser):
    parser.addoption(
        "--allow-unsafe-load",
        action="store_true",
        help="run tests that load pre-v2.4 model files with `weights_only=False`, which can "
        "execute arbitrary code. First, download a fresh copy of "
        "`tests/data/pre_v2_4_model_files.sha256` from GitHub, then check that the files are "
        "unmodified by running `sha256sum -c pre_v2_4_model_files.sha256` (on macOS, "
        "`shasum -a 256 -c pre_v2_4_model_files.sha256`) from `tests/data`",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--allow-unsafe-load"):
        return

    skip = pytest.mark.skip(
        reason="loads a pre-v2.4 model file with weights_only=False; pass --allow-unsafe-load to run"
    )
    for item in items:
        if item.get_closest_marker("unsafe_load") is not None:
            item.add_marker(skip)


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture
def pre_v2_4_model_path(request):
    """Get the path to a pre-v2.4 model file in `tests/data`, which must be loaded with
    `weights_only=False`. Raises an error if the test isn't marked with `unsafe_load`, so that
    tests can't load these files unless `--allow-unsafe-load` is passed."""

    def _pre_v2_4_model_path(relative_path: str) -> Path:
        if request.node.get_closest_marker("unsafe_load") is None:
            raise RuntimeError(
                f"'{relative_path}' must be loaded with `weights_only=False`, so tests that use it "
                "must be marked with `@pytest.mark.unsafe_load`."
            )

        return _DATA_DIR / relative_path

    return _pre_v2_4_model_path


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
