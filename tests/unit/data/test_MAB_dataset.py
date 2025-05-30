import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from chemprop.data.datasets import MolAtomBondDatapoint, MolAtomBondDataset
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.utils import make_mol


@pytest.fixture(params=[1, 5, 10])
def smis(smis, request):
    return smis.sample(request.param).to_list()


@pytest.fixture
def mols(smis):
    return [make_mol(smi, False, False) for smi in smis]


@pytest.fixture
def targets(mols):
    return [
        np.random.rand(len(mols), 2),
        [np.random.rand(mol.GetNumAtoms(), 2) for mol in mols],
        [np.random.rand(mol.GetNumBonds(), 2) for mol in mols],
    ]


@pytest.fixture
def X_d(mols):
    return [np.random.rand(2) for _ in mols]


@pytest.fixture
def V_fs(mols):
    return [np.random.rand(mol.GetNumAtoms(), 1) for mol in mols]


@pytest.fixture
def E_fs(mols):
    return [np.random.rand(mol.GetNumBonds(), 2) for mol in mols]


@pytest.fixture
def V_ds(mols):
    return [np.random.rand(mol.GetNumAtoms(), 3) for mol in mols]


@pytest.fixture
def E_ds(mols):
    return [np.random.rand(mol.GetNumBonds(), 4) for mol in mols]


@pytest.mark.parametrize(
    "X_d, V_fs, E_fs, V_ds, E_ds",
    [(None, None, None, None, None), ("X_d", "V_fs", "E_fs", "V_ds", "E_ds")],
    indirect=True,
)
@pytest.fixture
def data(mols, targets, X_d, V_fs, E_fs, V_ds, E_ds):
    mol_targets, atom_targets, bond_targets = targets
    return [
        MolAtomBondDatapoint(
            mol=mol,
            y=mol_targets[i],
            atom_y=atom_targets[i],
            bond_y=bond_targets[i],
            x_d=X_d[i] if X_d is not None else None,
            V_f=V_fs[i] if V_fs is not None else None,
            E_f=E_fs[i] if E_fs is not None else None,
            V_d=V_ds[i] if V_ds is not None else None,
            E_d=E_ds[i] if E_ds is not None else None,
        )
        for i, mol in enumerate(mols)
    ]


@pytest.fixture
def dataset(data):
    extra_atom_fdim = data[0].V_f.shape[1] if data[0].V_f is not None else 0
    extra_bond_fdim = data[0].E_f.shape[1] if data[0].E_f is not None else 0

    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=extra_atom_fdim, extra_bond_fdim=extra_bond_fdim
    )
    dset = MolAtomBondDataset(data, featurizer)
    return dset


def test_targets(dataset, targets):
    np.testing.assert_array_equal(dataset.Y, targets[0])
    for dset_Y, Y in zip(dataset.atom_Y, targets[1]):
        np.testing.assert_array_almost_equal(dset_Y, Y)
    for dset_Y, Y in zip(dataset.bond_Y, targets[2]):
        np.testing.assert_array_almost_equal(dset_Y, Y)


@pytest.mark.skipif(
    not all([x is None for x in ["X_d", "V_fs", "E_fs", "V_ds"]]), reason="Not all inputs are None"
)
def test_aux_nones(dataset: MolAtomBondDataset):
    np.testing.assert_array_equal(dataset.X_d, None)
    np.testing.assert_array_equal(dataset.V_fs, None)
    np.testing.assert_array_equal(dataset.E_fs, None)
    np.testing.assert_array_equal(dataset.V_ds, None)
    np.testing.assert_array_equal(dataset.E_ds, None)
    np.testing.assert_array_equal(dataset.atom_gt_mask, None)
    np.testing.assert_array_equal(dataset.bond_gt_mask, None)
    np.testing.assert_array_equal(dataset.atom_lt_mask, None)
    np.testing.assert_array_equal(dataset.bond_lt_mask, None)
    assert dataset.mol_dataset.d_xd == 0
    assert dataset.mol_dataset.d_vf == 0
    assert dataset.mol_dataset.d_ef == 0
    assert dataset.mol_dataset.d_vd == 0
    assert dataset.mol_dataset.d_ed == 0


def test_normalize_targets(dataset):
    dset_mol_scaler = dataset.normalize_targets()
    mol_scaler = StandardScaler()
    mol_scaler.fit(dataset._Y)
    Y = mol_scaler.transform(dataset._Y)

    np.testing.assert_array_equal(dataset.Y, Y)
    np.testing.assert_array_equal(dset_mol_scaler.mean_, mol_scaler.mean_)
    np.testing.assert_array_equal(dset_mol_scaler.scale_, mol_scaler.scale_)

    dset_atom_scaler = dataset.normalize_targets("atom")
    atom_scaler = StandardScaler()
    atom_scaler.fit(np.concatenate(dataset._atom_Y, axis=0))
    atom_Y = [atom_scaler.transform(y) for y in dataset._atom_Y]

    for dset_Y, Y in zip(dataset.atom_Y, atom_Y):
        np.testing.assert_array_equal(dset_Y, Y)
    np.testing.assert_array_equal(dset_atom_scaler.mean_, atom_scaler.mean_)
    np.testing.assert_array_equal(dset_atom_scaler.scale_, atom_scaler.scale_)

    dset_bond_scaler = dataset.normalize_targets("bond")
    bond_scaler = StandardScaler()
    bond_scaler.fit(np.concatenate(dataset._bond_Y, axis=0))
    bond_Y = [bond_scaler.transform(y) for y in dataset._bond_Y]

    for dset_Y, Y in zip(dataset.bond_Y, bond_Y):
        np.testing.assert_array_equal(dset_Y, Y)
    np.testing.assert_array_equal(dset_bond_scaler.mean_, bond_scaler.mean_)
    np.testing.assert_array_equal(dset_bond_scaler.scale_, bond_scaler.scale_)


def test_normalize_inputs(dataset):
    dset_scaler = dataset.normalize_inputs("E_d")
    scaler = StandardScaler()
    Xs = dataset._E_ds
    scaler.fit(np.concatenate(Xs, axis=0))
    scaled_E_ds = [scaler.transform(x) for x in Xs]

    for dset_X, X in zip(dataset.E_ds, scaled_E_ds):
        np.testing.assert_array_almost_equal(dset_X, X)
    np.testing.assert_array_equal(getattr(dset_scaler, "mean_"), scaler.mean_)
    np.testing.assert_array_equal(getattr(dset_scaler, "scale_"), scaler.scale_)
