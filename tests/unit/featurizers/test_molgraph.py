import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph import SimpleMoleculeMolGraphFeaturizer


@pytest.fixture(params=[0, 10, 100])
def extra(request):
    return request.param


@pytest.fixture
def atom_features_extra(mol, extra):
    n_a = mol.GetNumAtoms()

    return np.random.rand(n_a, extra)


@pytest.fixture
def bond_features_extra(mol, extra):
    n_b = mol.GetNumBonds()

    return np.random.rand(n_b, extra)


@pytest.fixture
def mol_featurizer():
    return SimpleMoleculeMolGraphFeaturizer()


@pytest.fixture
def mol_featurizer_extra(extra):
    return SimpleMoleculeMolGraphFeaturizer(None, None, extra, extra)


@pytest.fixture
def mg(mol, mol_featurizer):
    return mol_featurizer(mol)


def test_atom_fdim(extra):
    mf = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra)

    assert mf.atom_fdim == len(mf.atom_featurizer) + extra


def test_V_shape(mol, mol_featurizer: SimpleMoleculeMolGraphFeaturizer, mg: MolGraph):
    n_a = mol.GetNumAtoms()
    d_a = mol_featurizer.atom_fdim

    assert mg.V.shape == (n_a, d_a)


def test_E_shape(mol, mol_featurizer: SimpleMoleculeMolGraphFeaturizer, mg: MolGraph):
    n_b = mol.GetNumBonds()
    d_b = mol_featurizer.bond_fdim

    assert mg.E.shape == (2 * n_b, d_b)


def test_x2y_len(mol: Chem.Mol, mg: MolGraph):
    num_bonds = mol.GetNumBonds()

    assert mg.edge_index.shape == (2, 2 * num_bonds)
    assert mg.rev_edge_index.shape == (2 * num_bonds,)


def test_composability(mol):
    mf1 = SimpleMoleculeMolGraphFeaturizer(MultiHotAtomFeaturizer.v1(50))
    mf2 = SimpleMoleculeMolGraphFeaturizer(MultiHotAtomFeaturizer.v1(100))

    assert mf1(mol).V.shape != mf2(mol).V.shape


def test_invalid_atom_extra_shape(mol_featurizer, mol):
    n_a = mol.GetNumAtoms()
    with pytest.raises(ValueError):
        mol_featurizer(mol, atom_features_extra=np.random.rand(n_a + 1, 10))


def test_invalid_bond_extra_shape(mol_featurizer, mol):
    n_b = mol.GetNumBonds()
    with pytest.raises(ValueError):
        mol_featurizer(mol, bond_features_extra=np.random.rand(n_b + 1, 10))


def test_atom_extra_shape(mol, extra, atom_features_extra):
    mf = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra)
    mg = mf(mol, atom_features_extra=atom_features_extra)

    assert mg.V.shape == (mol.GetNumAtoms(), mf.atom_fdim)


def test_atom_extra_values(mol, extra, atom_features_extra):
    mf = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra)
    mg = mf(mol, atom_features_extra=atom_features_extra)

    np.testing.assert_array_equal(mg.V[:, len(mf.atom_featurizer) :], atom_features_extra)


def test_bond_extra(mol, extra, bond_features_extra):
    mf = SimpleMoleculeMolGraphFeaturizer(extra_bond_fdim=extra)
    mg = mf(mol, bond_features_extra=bond_features_extra)

    assert mg.E.shape == (2 * mol.GetNumBonds(), mf.bond_fdim)


def test_atom_bond_extra(mol, extra, atom_features_extra, bond_features_extra):
    mf = SimpleMoleculeMolGraphFeaturizer(extra_atom_fdim=extra, extra_bond_fdim=extra)
    mg = mf(mol, atom_features_extra, bond_features_extra)

    assert mg.E.shape == (2 * mol.GetNumBonds(), len(mf.bond_featurizer) + extra)
