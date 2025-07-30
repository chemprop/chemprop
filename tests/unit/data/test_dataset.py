from unittest.mock import MagicMock, call

import numpy as np
import pytest
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from chemprop.data.datapoints import LazyMoleculeDatapoint, MoleculeDatapoint
from chemprop.data.datasets import CuikmolmakerDataset, MoleculeDataset
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.molgraph.molecule import (
    CuikmolmakerMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.utils.utils import is_cuikmolmaker_available


@pytest.fixture(params=[1, 5, 10])
def smis(smis, request):
    return smis.sample(request.param).to_list()


@pytest.fixture
def targets(smis):
    return np.random.rand(len(smis), 1)


@pytest.fixture
def mols(smis):
    return [Chem.MolFromSmiles(smi) for smi in smis]


@pytest.fixture
def X_d(mols):
    return [np.random.rand(1) for _ in mols]


@pytest.fixture
def V_fs(mols):
    return [np.random.rand(mol.GetNumAtoms(), 1) for mol in mols]


@pytest.fixture
def E_fs(mols):
    return [np.random.rand(mol.GetNumBonds(), 2) for mol in mols]


@pytest.fixture
def V_ds(mols):
    return [np.random.rand(mol.GetNumAtoms(), 3) for mol in mols]


@pytest.mark.parametrize(
    "X_d, V_fs, E_fs, V_ds",
    [(None, None, None, None), ("X_d", "V_fs", "E_fs", "V_ds")],
    indirect=True,
)
@pytest.fixture
def data(mols, targets, X_d, V_fs, E_fs, V_ds):
    return [
        MoleculeDatapoint(mol=mol, y=target, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for mol, target, x_d, V_f, E_f, V_d in zip(mols, targets, X_d, V_fs, E_fs, V_ds)
    ]


@pytest.mark.parametrize(
    "X_d, V_fs, E_fs, V_ds",
    [(None, None, None, None), ("X_d", "V_fs", "E_fs", "V_ds")],
    indirect=True,
)
@pytest.fixture
def lazy_data(smis, targets, X_d, V_fs, E_fs, V_ds):
    return [
        LazyMoleculeDatapoint(smi, y=target, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for smi, target, x_d, V_f, E_f, V_d in zip(smis, targets, X_d, V_fs, E_fs, V_ds)
    ]


@pytest.fixture(params=[False, True])
def cache(request):
    return request.param


@pytest.fixture
def dataset(data, cache):
    extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
    extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0

    dset = MoleculeDataset(
        data,
        SimpleMoleculeMolGraphFeaturizer(
            extra_atom_fdim=extra_atom_fdim, extra_bond_fdim=extra_bond_fdim
        ),
    )
    dset.cache = cache

    return dset


@pytest.fixture
def cuik_molecule_dataset(lazy_data):
    if is_cuikmolmaker_available():
        featurizer = CuikmolmakerMolGraphFeaturizer(atom_featurizer_mode="V2")
        dset = CuikmolmakerDataset(lazy_data, featurizer)
        return dset
    else:
        return None


def test_none():
    with pytest.raises(ValueError):
        MoleculeDataset(None, SimpleMoleculeMolGraphFeaturizer())


def test_empty():
    """TODO"""


def test_len(data, dataset, cuik_molecule_dataset):
    assert len(data) == len(dataset)
    if is_cuikmolmaker_available():
        assert len(data) == len(cuik_molecule_dataset)


def test_smis(dataset, cuik_molecule_dataset, smis):
    assert smis == dataset.smiles
    if is_cuikmolmaker_available():
        assert smis == cuik_molecule_dataset.smiles


def test_targets(dataset, cuik_molecule_dataset, targets):
    np.testing.assert_array_equal(dataset.Y, targets)
    if is_cuikmolmaker_available():
        np.testing.assert_array_equal(cuik_molecule_dataset.Y, targets)


def test_set_targets_too_short(dataset):
    with pytest.raises(ValueError):
        dataset.Y = np.random.rand(len(dataset) // 2, 1)


def test_num_tasks(dataset, cuik_molecule_dataset, targets):
    assert dataset.t == targets.shape[1]
    if is_cuikmolmaker_available():
        assert cuik_molecule_dataset.t == targets.shape[1]


#  __getitem__ is slower than __getitems__ for CuikmolmakerDataset, so we don't use it in our
# examples, but we still define it for completeness, so test it here.
@pytest.mark.skipif(not is_cuikmolmaker_available(), reason="cuik_molmaker not installed")
def test_getitem(cuik_molecule_dataset):
    datum = cuik_molecule_dataset[0]
    assert datum is not None


@pytest.mark.skipif(
    not all([x is None for x in ["X_d", "V_fs", "E_fs", "V_ds"]]), reason="Not all inputs are None"
)
def test_aux_nones(dataset: MoleculeDataset):
    np.testing.assert_array_equal(dataset.X_d, None)
    np.testing.assert_array_equal(dataset.V_fs, None)
    np.testing.assert_array_equal(dataset.E_fs, None)
    np.testing.assert_array_equal(dataset.V_ds, None)
    np.testing.assert_array_equal(dataset.gt_mask, None)
    np.testing.assert_array_equal(dataset.lt_mask, None)
    assert dataset.d_xd == 0
    assert dataset.d_vf == 0
    assert dataset.d_ef == 0
    assert dataset.d_vd == 0


def test_normalize_targets(dataset):
    dset_scaler = dataset.normalize_targets()
    scaler = StandardScaler()
    scaler.fit(dataset._Y)
    Y = scaler.transform(dataset._Y)

    np.testing.assert_array_equal(dataset.Y, Y)
    np.testing.assert_array_equal(dset_scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(dset_scaler.scale_, scaler.scale_)


def test_normalize_inputs(dataset):
    dset_scaler = dataset.normalize_inputs("X_d")
    scaler = StandardScaler()
    scaler.fit(dataset._X_d)
    X = scaler.transform(dataset._X_d)

    np.testing.assert_array_equal(dataset.X_d, X)
    np.testing.assert_array_equal(dset_scaler.mean_, scaler.mean_)
    np.testing.assert_array_equal(dset_scaler.scale_, scaler.scale_)

    inputs = ["V_f", "E_f", "V_d"]
    for input_ in inputs:
        dset_scaler = dataset.normalize_inputs(input_)
        scaler = StandardScaler()
        Xs = getattr(dataset, f"_{input_}s")
        X = np.concatenate(Xs, axis=0)
        scaler.fit(X)
        Xs = [scaler.transform(x) for x in Xs]

        for X, dset_X in zip(Xs, getattr(dataset, f"{input_}s")):
            np.testing.assert_array_equal(X, dset_X)
        np.testing.assert_array_equal(getattr(dset_scaler, "mean_"), scaler.mean_)
        np.testing.assert_array_equal(getattr(dset_scaler, "scale_"), scaler.scale_)


@pytest.mark.parametrize("cache", [False, True])
def test_cache(dataset: MoleculeDataset, cache):
    """Test that cache attribute is being set appropriately and that the underlying cache is being
    used correctly to load the molgraphs."""
    mg = MolGraph(None, None, None, None)

    dataset.cache = cache
    assert dataset.cache == cache
    dataset.mg_cache = MagicMock()
    dataset.mg_cache.__getitem__.side_effect = lambda i: mg

    calls = []
    for i in range(len(dataset)):
        assert dataset[i].mg is mg
        calls.append(call(i))

    dataset.mg_cache.__getitem__.assert_has_calls(calls)
