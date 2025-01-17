import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.datasets import (
    AtomDataset,
    BondDataset,
    MolAtomBondDataset,
    MoleculeDatapoint,
    MoleculeDataset,
)
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer
from chemprop.utils import make_mol


@pytest.fixture(params=[1, 5, 10])
def smis(smis, request):
    return smis.sample(request.param).to_list()


@pytest.fixture
def targets(smis):
    num_atoms, num_bonds = 0, 0
    for i in range(len(smis)):
        num_atoms += make_mol(smis[i], False, False).GetNumAtoms()
        num_bonds += make_mol(smis[i], False, False).GetNumBonds()
    return [
        np.random.rand(len(smis), 1),
        np.random.rand(num_atoms, 1),
        np.random.rand(num_bonds, 1),
    ]


@pytest.fixture
def slices(smis):
    atom_slices, bond_slices = [], []
    atom_slices.append(0)
    bond_slices.append(0)
    for i in range(len(smis)):
        atom_slices.append(atom_slices[i] + make_mol(smis[i], False, False).GetNumAtoms())
        bond_slices.append(bond_slices[i] + make_mol(smis[i], False, False).GetNumBonds())
    return [atom_slices, bond_slices]


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


@pytest.fixture
def E_ds(mols):
    return [np.random.rand(mol.GetNumBonds(), 4) for mol in mols]


@pytest.mark.parametrize(
    "X_d, V_fs, E_fs, V_ds, E_ds",
    [(None, None, None, None, None), ("X_d", "V_fs", "E_fs", "V_ds", "E_ds")],
    indirect=True,
)
@pytest.fixture
def data(mols, targets, X_d, V_fs, E_fs, V_ds, E_ds, slices):
    mol_data = [
        MoleculeDatapoint(mol=mol, y=target, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d, E_d=E_d)
        for mol, target, x_d, V_f, E_f, V_d, E_d in zip(
            mols, targets[0], X_d, V_fs, E_fs, V_ds, E_ds
        )
    ]
    atom_targets, bond_targets = [], []
    for i in range(len(slices[0]) - 1):
        atom_targets.append(targets[1][slices[0][i] : slices[0][i + 1]])
    for i in range(len(slices[1]) - 1):
        bond_targets.append(targets[2][slices[1][i] : slices[1][i + 1]])
    atom_data = [
        MoleculeDatapoint(mol=mol, y=target, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d)
        for mol, target, x_d, V_f, E_f, V_d, E_d in zip(
            mols, atom_targets, X_d, V_fs, E_fs, V_ds, E_ds
        )
    ]
    bond_data = [
        MoleculeDatapoint(mol=mol, y=target, x_d=x_d, V_f=V_f, E_f=E_f, V_d=V_d, E_d=E_d)
        for mol, target, x_d, V_f, E_f, V_d, E_d in zip(
            mols, bond_targets, X_d, V_fs, E_fs, V_ds, E_ds
        )
    ]
    return [mol_data, atom_data, bond_data]


@pytest.fixture(params=[False, True])
def cache(request):
    return request.param


@pytest.fixture
def dataset(data, cache):
    extra_atom_fdim = data[0][0].V_f.shape[1] if data[0][0].V_f is not None else 0
    extra_bond_fdim = data[0][0].E_f.shape[1] if data[0][0].E_f is not None else 0

    featurizer = SimpleMoleculeMolGraphFeaturizer(
        extra_atom_fdim=extra_atom_fdim, extra_bond_fdim=extra_bond_fdim
    )
    mol_dset = MoleculeDataset(data[0], featurizer)
    atom_dset = AtomDataset(data[1], featurizer)
    bond_dset = BondDataset(data[2], featurizer)
    dset = MolAtomBondDataset(mol_dset, atom_dset, bond_dset)
    return dset


def test_none():
    with pytest.raises(ValueError):
        MoleculeDataset(None, SimpleMoleculeMolGraphFeaturizer())


def test_empty():
    """TODO"""


def test_len(data, dataset):
    assert len(data[0]) == len(dataset)


def test_smis(dataset, smis):
    assert smis == dataset.mol_dataset.smiles


def test_targets(dataset, targets):
    np.testing.assert_array_equal(dataset.mol_dataset.Y, targets[0])
    np.testing.assert_array_equal(dataset.atom_dataset.Y, targets[1])
    np.testing.assert_array_equal(dataset.bond_dataset.Y, targets[2])


@pytest.mark.skipif(
    not all([x is None for x in ["X_d", "V_fs", "E_fs", "V_ds"]]), reason="Not all inputs are None"
)
def test_aux_nones(dataset: MoleculeDataset):
    np.testing.assert_array_equal(dataset.mol_dataset.X_d, None)
    np.testing.assert_array_equal(dataset.mol_dataset.V_fs, None)
    np.testing.assert_array_equal(dataset.mol_dataset.E_fs, None)
    np.testing.assert_array_equal(dataset.mol_dataset.V_ds, None)
    np.testing.assert_array_equal(dataset.mol_dataset.gt_mask, None)
    np.testing.assert_array_equal(dataset.mol_dataset.lt_mask, None)
    assert dataset.mol_dataset.d_xd == 0
    assert dataset.mol_dataset.d_vf == 0
    assert dataset.mol_dataset.d_ef == 0
    assert dataset.mol_dataset.d_vd == 0
