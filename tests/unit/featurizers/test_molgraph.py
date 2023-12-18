import numpy as np
import pytest
from rdkit import Chem

from chemprop.featurizers import (
    MoleculeMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
    MolGraph,
    MultiHotAtomFeaturizer,
)


@pytest.fixture(
    params=[
        "Fc1cccc(C2(c3nnc(Cc4cccc5ccccc45)o3)CCOCC2)c1",
        "O=C(NCc1ccnc(Oc2ccc(F)cc2)c1)c1[nH]nc2c1CCCC2",
        "Cc1ccccc1CC(=O)N1CCN(CC(=O)N2Cc3ccccc3C(c3ccccc3)C2)CC1",
        "O=C(Nc1cc2c(cn1)CCCC2)N1CCCC1c1ccc(O)cc1",
        "NC(=O)C1CCN(C(=O)CCc2c(-c3ccc(F)cc3)[nH]c3ccccc23)C1",
    ]
)
def smi(request):
    return request.param


@pytest.fixture
def mol(smi):
    return Chem.MolFromSmiles(smi)


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


def test_abc():
    with pytest.raises(TypeError):
        MoleculeMolGraphFeaturizer()


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
    mf1 = SimpleMoleculeMolGraphFeaturizer(MultiHotAtomFeaturizer(50))
    mf2 = SimpleMoleculeMolGraphFeaturizer(MultiHotAtomFeaturizer(100))

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
