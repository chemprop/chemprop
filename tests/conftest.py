import ast
import hashlib
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
# `weights_only=False`, which can execute arbitrary code. Their hashes are checked before they are
# loaded to ensure they haven't been changed.
PRE_V2_4_MODEL_SHA256 = {
    "v1_model_files/example_model_v1_4.pt": "ff25a9e3bb73ef6e659fd030f9c025050ae19a073daaa2cf5d5b50bcdb1bd05a",
    "v1_model_files/example_model_v1_regression_mol.pt": "5dad80688a828c46861df5c8b1100c4d4ea030eec28aab906577c8dd70f1fb4b",
    "pre_v2_4_model_files/example_model_v2_classification_dirichlet_mol.pt": "aff5fab340eabe94c131880eb3292dc8737f4a2de0ebf7c492b43bf07aa935fc",
    "pre_v2_4_model_files/example_model_v2_classification_mol.pt": "1f6d13a6094d197f543d32e502991fc5af4b37d92dfe03c8b6eef11c9b5c61ed",
    "pre_v2_4_model_files/example_model_v2_classification_mol_multiclass.pt": "845c7960c3c0245b8f7de946c2f90d4e7704860a53001cc31be3d031cfeb70af",
    "pre_v2_4_model_files/example_model_v2_classification_mol_with_metrics.ckpt": "6ad46d783130249352d3da372a250ba2baca8a655f69c47ea7d6c9f183af8c43",
    "pre_v2_4_model_files/example_model_v2_multiclass_dirichlet_mol.pt": "2301dd7918851f19022938716c15501c9397e0b2f629753c0b7c03349851b21b",
    "pre_v2_4_model_files/example_model_v2_regression_evidential_mol.pt": "1c056822e6e83801b7f4c89693df5404123a105142ece80737096be19a065f06",
    "pre_v2_4_model_files/example_model_v2_regression_mol+mol+rxn_check_predictions.pt": "61ea495f85e1a5f9321b4610c82d84cac9909f03e563b38af115c6efd40a785a",
    "pre_v2_4_model_files/example_model_v2_regression_mol+mol.ckpt": "e0314c3a1c8971ee55e6d5cf7fb90b867b861ea1950087701d7d1761a7a96314",
    "pre_v2_4_model_files/example_model_v2_regression_mol+mol.pt": "e8a0a992770e486982f5ff7b24425886a6b7ffba92d975fc0840185d97b03da4",
    "pre_v2_4_model_files/example_model_v2_regression_mol.ckpt": "ec996c623cea37b6690f13c6e41560937eb1768c8b180a41e97b03526b12fd1b",
    "pre_v2_4_model_files/example_model_v2_regression_mol.pt": "7a222b920c9152d67f6e1970881221079ab8a321d2db57759921da51b5b4a825",
    "pre_v2_4_model_files/example_model_v2_regression_mol_multitask.pt": "9d053982d9a9e524d9b30aa2df76f38f07d0752663515fc997d4824305b5b332",
    "pre_v2_4_model_files/example_model_v2_regression_mol_with_metrics.ckpt": "18ac4e7f54c39128cbde9d3f04312af57db6b1ffe2e9f6154aceeca42964c81d",
    "pre_v2_4_model_files/example_model_v2_regression_mve_mol.pt": "2b95ee7e06ff51b29868badb2f548bb24d458f351a979a0393321d003fa1ad15",
    "pre_v2_4_model_files/example_model_v2_regression_quantile_mol.pt": "b034fa1527f5c43da727514309a15be175affb2786488b93c9978f39377f57cc",
    "pre_v2_4_model_files/example_model_v2_regression_rxn+mol.pt": "10908249c6ed837dab533ddb9b1613a229198f6210bdf8dc9b0ad9dbba4bce33",
    "pre_v2_4_model_files/example_model_v2_regression_rxn.ckpt": "63289026b497e50517ba7b89aeb892bd0f6a0a2aaeea8053fd2a23c5c7eec892",
    "pre_v2_4_model_files/example_model_v2_regression_rxn.pt": "2aa792943e8575599b1d3af31667bcd8ec8961a04a6deb11d9ffe0e9cbcce278",
    "pre_v2_4_model_files/example_model_v2_trained_on_cuda.pt": "a1db40c0b5fae241ec8697cd066f0e3fa7384086db8f207f144229fb6be44241",
    "pre_v2_4_model_files/mol_atom_bond/QM_descriptors.pt": "45831541ca42ab7400abe68b85ec38a889b748bb3b46950611def376bfc5395b",
    "pre_v2_4_model_files/mol_atom_bond/atomic_regression_atom_mapped.pt": "4c1de6a433477fe5ff9422e6adf61317cf522273b1322f42612152b1d2bfe26a",
    "pre_v2_4_model_files/mol_atom_bond/classification.pt": "63b016824bf62ad403d6ad40e7186ad0a9e8b609aabe54b64c9d2147ea7559d1",
    "pre_v2_4_model_files/mol_atom_bond/multiclass.pt": "a6ff7e4db61c8b9958b58926a432ef5d7e9530a25885bd3b4d706c9ca248c34e",
    "pre_v2_4_model_files/mol_atom_bond/regression.pt": "de1fa978876a6a205008a095771e57686e19e874c8f3dcd2fccf52d3b34a5b4f",
    "pre_v2_4_model_files/mol_atom_bond/regression_constrained.pt": "676b329a5a7801f962a87c544f12b8fe2927abc65694abd8b7de61735fc18c00",
    "pre_v2_4_model_files/mol_atom_bond/regression_mve.pt": "69b34b40bb4b3df5da810c7d2ee42fbc7677d15f7b4f43c42ee4d4af141ba6fc",
    "pre_v2_4_model_files/mol_atom_bond/regression_no_atom.pt": "d158caa5320ffba286aea6bb25575722a42ef3b5e418879238964d6c7ff9cd41",
    "pre_v2_4_model_files/mol_atom_bond/regression_no_bond.pt": "53ddf5dca30d6663931a57166382ec7f0403df57af7a30f51c15842d3f20a8a6",
    "pre_v2_4_model_files/mol_atom_bond/regression_no_mol.pt": "9a3a18c56860ca67081fc7150033283c6ad468d3a7c2bdc5b6fb243ac1240819",
    "pre_v2_4_model_files/mol_atom_bond/regression_only_atom.pt": "5168f06a47bc29e2beda206bd80a4531846ae01c441740edcf2f6c96f3db59ed",
    "pre_v2_4_model_files/mol_atom_bond/regression_only_bond.pt": "83e4e551ca69e81a2a4cb5da24ee179aa460dd915169fdaeea5c92bfdfae7415",
    "pre_v2_4_model_files/mol_atom_bond/regression_only_mol.pt": "dfbe7cc17161f922e6a2613ebba8439d87e23f11c56a9f98e023292b153bdca0",
    "pre_v2_4_model_files/mol_atom_bond/regression_with_extras.pt": "23ce8a133c373a117004aa43311dccb3b844ff6fe48d5bad84825d4221a6f5fb",
}


@pytest.fixture
def data_dir():
    return _DATA_DIR


@pytest.fixture
def pre_v2_4_model_path():
    """Get the path to a pre-v2.4 model file in `tests/data`, checking that its hash is unchanged"""

    def _pre_v2_4_model_path(relative_path: str) -> Path:
        path = _DATA_DIR / relative_path
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha256 != PRE_V2_4_MODEL_SHA256.get(relative_path):
            raise ValueError(
                f"The hash of '{path}' doesn't match the expected hash! This file is loaded with "
                "`weights_only=False`, so it must not be changed."
            )

        return path

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
