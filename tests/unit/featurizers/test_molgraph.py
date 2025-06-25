import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.molgraph import MolGraph
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph import (
    CuikmolmakerMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.utils.utils import is_cuikmolmaker_available


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


@pytest.fixture(params=["V2", "V1", "ORGANIC", "RIGR"])
def batch_mol_featurizer(request):
    if is_cuikmolmaker_available():
        return CuikmolmakerMolGraphFeaturizer(atom_featurizer_mode=request.param)
    else:
        return None


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


@pytest.mark.skipif(not is_cuikmolmaker_available(), reason="cuik_molmaker not installed")
def test_batch_V_shape(smis, mols, batch_mol_featurizer: CuikmolmakerMolGraphFeaturizer):
    total_num_atoms = sum([mol.GetNumAtoms() for mol in mols])
    batch = batch_mol_featurizer(smis)
    batch_V = batch[0]
    assert batch_V.shape[0] == total_num_atoms
    if batch_mol_featurizer.atom_featurizer_mode == "V1":
        assert batch_V.shape[1] == 133
    elif batch_mol_featurizer.atom_featurizer_mode == "V2":
        assert batch_V.shape[1] == 72
    elif batch_mol_featurizer.atom_featurizer_mode == "ORGANIC":
        assert batch_V.shape[1] == 44
    elif batch_mol_featurizer.atom_featurizer_mode == "RIGR":
        assert batch_V.shape[1] == 53


def test_E_shape(mol, mol_featurizer: SimpleMoleculeMolGraphFeaturizer, mg: MolGraph):
    n_b = mol.GetNumBonds()
    d_b = mol_featurizer.bond_fdim

    assert mg.E.shape == (2 * n_b, d_b)


@pytest.mark.skipif(not is_cuikmolmaker_available(), reason="cuik_molmaker not installed")
def test_batch_E_shape(smis, mols, batch_mol_featurizer: CuikmolmakerMolGraphFeaturizer):
    total_num_bonds = sum([mol.GetNumBonds() for mol in mols])
    batch = batch_mol_featurizer(smis)
    batch_E = batch[1]
    assert batch_E.shape[0] == 2 * total_num_bonds
    if batch_mol_featurizer.atom_featurizer_mode == "RIGR":
        assert batch_E.shape[1] == 2
    else:
        assert batch_E.shape[1] == 14


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
