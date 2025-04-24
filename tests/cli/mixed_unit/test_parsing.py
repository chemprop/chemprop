from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chemprop.cli.utils import build_mixed_data_from_files, make_dataset
from chemprop.data import MolAtomBondDataset

_DATA_PATH = Path(__file__).parent.parent.parent / "data/regression/mixed/mixed.csv"


@pytest.fixture
def data_path():
    return _DATA_PATH


@pytest.fixture
def format_kwargs(data_path):
    df = pd.read_csv(data_path)
    format_kwargs = dict(
        no_header_row=False,
        smiles_cols=["smiles"],
        rxn_cols=None,
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=False,
    )
    format_kwargs["target_cols"] = df.columns[1:].tolist()
    return format_kwargs


@pytest.fixture
def data(data_path, format_kwargs):
    featurization_kwargs = dict(molecule_featurizers=None, keep_h=False, add_h=False)
    datas, mol_cols, atom_cols, bond_cols = build_mixed_data_from_files(
        data_path,
        **format_kwargs,
        p_descriptors=None,
        p_atom_feats=None,
        p_bond_feats=None,
        p_atom_descs=None,
        **featurization_kwargs,
    )
    return [datas, mol_cols, atom_cols, bond_cols]


def test_dataset(data):
    dset = [make_dataset(d, "REAC_DIFF", "V2") for d in data[0]]
    assert isinstance(dset[0], MolAtomBondDataset)
    print(dset)
    assert dset[0].data[0].y == np.array([1])
    assert (dset[0].data[0].atom_y == np.array([[1], [2]])).all()
    assert dset[0].data[0].bond_y == np.array([[3]])


def test_data(data_path, data):
    df = pd.read_csv(data_path)
    assert len(data[1]) + len(data[2]) + len(data[3]) == df.shape[1] - 1
