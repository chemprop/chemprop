import numpy as np
import pytest
from rdkit import Chem

from chemprop.v2.featurizers import MolGraphFeaturizerProto, MoleculeMolGraphFeaturizer, MolGraph, AtomFeaturizer


@pytest.fixture(params=[
    'Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1',
    'O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2',
    'Cc1ccccc1CC(=O)N1CCN(CC(=O)N2Cc3ccccc3C(c3ccccc3)C2)CC1',
    'O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1',
    'NC(=O)C1CCN(C(=O)CCc2c(-c3ccc(F)cc3)[nH]c3ccccc23)C1',
])
def smi(request):
    return request.param


@pytest.fixture
def mol(smi):
    return Chem.MolFromSmiles(smi)


@pytest.fixture(params=[True, False])
def atom_messages(request):
    return request.param


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
def mol_featurizer(atom_messages):
    return MoleculeMolGraphFeaturizer(bond_messages=not atom_messages)


@pytest.fixture
def mol_featurizer_extra(extra):
    return MoleculeMolGraphFeaturizer(None, None, extra, extra, False)


@pytest.fixture
def mg(mol, mol_featurizer):
    return mol_featurizer(mol)


def test_abc():
    with pytest.raises(TypeError):
        MolGraphFeaturizerProto()
    

def test_atom_fdim(extra):
    mf = MoleculeMolGraphFeaturizer(extra_atom_fdim=extra)

    assert mf.atom_fdim == len(mf.atom_featurizer) + extra


def test_bond_fdim(atom_messages, extra):
    mf = MoleculeMolGraphFeaturizer(extra_bond_fdim=extra, bond_messages=atom_messages)

    if atom_messages:
        assert mf.bond_fdim == len(mf.bond_featurizer) + extra
    else:
        assert mf.bond_fdim == len(mf.bond_featurizer) + extra + mf.atom_fdim


def test_n_atoms_bonds(mol: Chem.Mol, mg: MolGraph):
    assert mol.GetNumAtoms() == mg.n_atoms
    assert 2 * mol.GetNumBonds() == mg.n_bonds


def test_X_v_shape(mol, mol_featurizer: MoleculeMolGraphFeaturizer, mg: MolGraph):
    n_a = mol.GetNumAtoms()
    d_a = mol_featurizer.atom_fdim

    assert mg.V.shape == (n_a, d_a)


def test_X_e_shape(mol, mol_featurizer: MoleculeMolGraphFeaturizer, mg: MolGraph):
    n_b = mol.GetNumBonds()
    d_b = mol_featurizer.bond_fdim

    assert mg.E.shape == (2 * n_b, d_b)


def test_x2y_len(mol: Chem.Mol, mg: MolGraph):
    n_a = mol.GetNumAtoms()
    n_b = mol.GetNumBonds()
    
    assert len(mg.a2b) == n_a
    assert len(mg.b2a) == 2 * n_b
    assert len(mg.b2revb) == 2 * n_b


def test_a2a_none(mg):
    assert mg.a2a is None


def test_b2b_none(mg):
    assert mg.b2b is None


def test_composability(mol):
    mf1 = MoleculeMolGraphFeaturizer(AtomFeaturizer(50))
    mf2 = MoleculeMolGraphFeaturizer(AtomFeaturizer(100))

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
    mf = MoleculeMolGraphFeaturizer(None, None, extra)
    mg = mf(mol, atom_features_extra=atom_features_extra)

    assert mg.V.shape == (mol.GetNumAtoms(), mf.atom_fdim)


def test_atom_extra_values(mol, extra, atom_features_extra):
    mf = MoleculeMolGraphFeaturizer(None, None, extra)
    mg = mf(mol, atom_features_extra=atom_features_extra)

    np.testing.assert_array_equal(mg.V[:, len(mf.atom_featurizer):], atom_features_extra)


def test_bond_extra(mol, extra, bond_features_extra):
    mf = MoleculeMolGraphFeaturizer(None, None, 0, extra)
    mg = mf(mol, bond_features_extra=bond_features_extra)

    assert mg.E.shape == (2 * mol.GetNumBonds(), mf.bond_fdim)


def test_atom_bond_extra(mol, extra, atom_features_extra, bond_features_extra):
    mf = MoleculeMolGraphFeaturizer(None, None, extra, extra, False)
    mg = mf(mol, atom_features_extra, bond_features_extra)

    assert mg.E.shape == (
        2 * mol.GetNumBonds(), len(mf.atom_featurizer) + len(mf.bond_featurizer) + 2 * extra
    )
