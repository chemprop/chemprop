from pathlib import Path

import pandas as pd
import pytest

from chemprop.cli.utils import build_mixed_data_from_files, make_dataset
from chemprop.data import MockDataset, MolAtomBondDataset

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
    dsets = [
        make_dataset(data[0][d], "REAC_DIFF", "V2", d) if data[d + 1] else MockDataset()
        for d in range(len(data[0]))
    ]
    dset = MolAtomBondDataset(dsets[0], dsets[1], dsets[2])
    assert dset.mol_dataset == dsets[0]
    assert dset.atom_dataset == dsets[1]
    assert dset.bond_dataset == dsets[2]


def test_data(data_path, data):
    df = pd.read_csv(data_path)
    assert len(data[1]) + len(data[2]) + len(data[3]) == df.shape[1] - 1
    assert len(data[0]) == 3
