import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.datapoints import PolymerDatapoint
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.molgraph import PolymerMolGraphFeaturizer


@pytest.fixture(params=[0, 10, 100])
def extra(request):
    return request.param


@pytest.fixture
def atom_features_extra(polymer_mol, extra):
    n_a = polymer_mol.GetNumAtoms()

    return np.random.rand(n_a, extra)


@pytest.fixture
def bond_features_extra(polymer, polymer_mol, extra):
    n_b = polymer_mol.GetNumBonds()
    p_b = len(polymer.edges)

    return np.random.rand(n_b + p_b, extra)


@pytest.fixture
def polymer_smiles():
    return "[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1c(O)cc(O)c([*:4])c1O|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5~10"


@pytest.fixture
def polymer(polymer_smiles):
    return PolymerDatapoint.from_smi(polymer_smiles)


@pytest.fixture
def polymer_featurizer():
    return PolymerMolGraphFeaturizer()


@pytest.fixture
def featurized_polymer(polymer, polymer_featurizer):
    return polymer_featurizer(polymer)


@pytest.fixture
def polymer_mol(polymer):
    rwmol = Chem.rdchem.RWMol(polymer.mol)
    indicies = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    while len(indicies) > 0:
        rwmol.RemoveAtom(indicies[0])
        indicies = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)

    return rwmol


@pytest.fixture
def polymer_V_w():
    return np.array(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )


@pytest.fixture
def polymer_E_w():
    return np.array(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )


def test_fragment_weights(polymer, polymer_smiles):
    weights = polymer_smiles.split("|")[1:-1]

    assert polymer.fragment_weights == weights


def test_edges(polymer, polymer_smiles):
    edges = polymer_smiles.split("<")[1:]

    assert len(polymer.edges) == len(edges)
    assert polymer.edges == edges


def test_degree_of_polymerization(featurized_polymer, polymer_smiles):
    degree_of_poly = float(polymer_smiles.split("~")[1])

    assert featurized_polymer.degree_of_poly == (1.0 + np.log10(degree_of_poly))


def test_atom_fdim(polymer_featurizer):
    assert polymer_featurizer.atom_fdim == len(polymer_featurizer.atom_featurizer)


def test_V_shape(polymer_mol, polymer_featurizer, featurized_polymer):
    n_a = polymer_mol.GetNumAtoms()
    d_a = polymer_featurizer.atom_fdim

    assert featurized_polymer.V.shape == (n_a, d_a)


def test_E_shape(polymer, polymer_mol, polymer_featurizer, featurized_polymer):
    n_b = polymer_mol.GetNumBonds()
    p_b = len(polymer.edges) * 2
    d_b = polymer_featurizer.bond_fdim

    assert featurized_polymer.E.shape == (2 * n_b + p_b, d_b)


def test_V_w_shape(polymer_mol, featurized_polymer):
    n_a = polymer_mol.GetNumAtoms()

    assert featurized_polymer.V_w.shape == (n_a,)


def test_V_w(featurized_polymer, polymer_V_w):
    np.testing.assert_array_almost_equal(featurized_polymer.V_w, polymer_V_w)


def test_E_w_shape(polymer, polymer_mol, featurized_polymer):
    n_b = polymer_mol.GetNumBonds()
    p_b = len(polymer.edges) * 2

    assert featurized_polymer.E_w.shape == (2 * n_b + p_b,)


def test_E_w(featurized_polymer, polymer_E_w):
    np.testing.assert_array_almost_equal(featurized_polymer.E_w, polymer_E_w)


def test_x2y_len(polymer, polymer_mol, featurized_polymer):
    num_bonds = polymer_mol.GetNumBonds()
    p_b = len(polymer.edges) * 2

    assert featurized_polymer.edge_index.shape == (2, 2 * num_bonds + p_b)
    assert featurized_polymer.rev_edge_index.shape == (2 * num_bonds + p_b,)


def test_composability(polymer):
    mf1 = PolymerMolGraphFeaturizer(MultiHotAtomFeaturizer.v1(50))
    mf2 = PolymerMolGraphFeaturizer(MultiHotAtomFeaturizer.v1(100))

    assert mf1(polymer).V.shape != mf2(polymer).V.shape


def test_invalid_atom_extra_shape(polymer_featurizer, polymer_mol, polymer):
    n_a = polymer_mol.GetNumAtoms()
    with pytest.raises(ValueError):
        polymer_featurizer(polymer, atom_features_extra=np.random.rand(n_a + 1, 10))


def test_invalid_bond_extra_shape(polymer_featurizer, polymer_mol, polymer):
    n_b = polymer_mol.GetNumBonds()
    with pytest.raises(ValueError):
        polymer_featurizer(polymer, bond_features_extra=np.random.rand(n_b + 1, 10))


def test_atom_extra_shape(polymer, polymer_mol, extra, atom_features_extra):
    mf = PolymerMolGraphFeaturizer(extra_atom_fdim=extra)
    mg = mf(polymer, atom_features_extra=atom_features_extra)

    assert mg.V.shape == (polymer_mol.GetNumAtoms(), mf.atom_fdim)


def test_atom_extra_values(polymer, extra, atom_features_extra):
    mf = PolymerMolGraphFeaturizer(extra_atom_fdim=extra)
    mg = mf(polymer, atom_features_extra=atom_features_extra)

    np.testing.assert_array_equal(mg.V[:, len(mf.atom_featurizer) :], atom_features_extra)


def test_bond_extra(polymer, polymer_mol, extra, bond_features_extra):
    mf = PolymerMolGraphFeaturizer(extra_bond_fdim=extra)
    mg = mf(polymer, bond_features_extra=bond_features_extra)

    assert mg.E.shape == (2 * polymer_mol.GetNumBonds() + 2 * len(polymer.edges), mf.bond_fdim)


def test_atom_bond_extra(polymer, polymer_mol, extra, atom_features_extra, bond_features_extra):
    mf = PolymerMolGraphFeaturizer(extra_atom_fdim=extra, extra_bond_fdim=extra)
    mg = mf(polymer, atom_features_extra, bond_features_extra)

    assert mg.E.shape == (
        2 * polymer_mol.GetNumBonds() + 2 * len(polymer.edges),
        len(mf.bond_featurizer) + extra,
    )
