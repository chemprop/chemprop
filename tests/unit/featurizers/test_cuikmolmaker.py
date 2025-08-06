import numpy as np
import pytest

from chemprop.data.collate import BatchMolGraph
from chemprop.featurizers.atom import (
    MultiHotAtomFeaturizer,
    RIGRAtomFeaturizer,
    get_multi_hot_atom_featurizer,
)
from chemprop.featurizers.bond import MultiHotBondFeaturizer, RIGRBondFeaturizer
from chemprop.featurizers.molgraph import (
    CuikmolmakerMolGraphFeaturizer,
    SimpleMoleculeMolGraphFeaturizer,
)
from chemprop.utils.utils import is_cuikmolmaker_available

if not is_cuikmolmaker_available():
    pytest.skip("cuik_molmaker not installed", allow_module_level=True)


@pytest.fixture(params=[0, 10])
def extra(request):
    return request.param


@pytest.fixture(params=["V2", "V1", "ORGANIC", "RIGR"])
def atom_featurizer_mode(request):
    return request.param


@pytest.fixture
def num_atoms(mols):
    return sum(mol.GetNumAtoms() for mol in mols)


@pytest.fixture
def num_bonds(mols):
    return sum(mol.GetNumBonds() for mol in mols)


@pytest.fixture
def atom_features_extra(num_atoms, extra):
    return np.random.rand(num_atoms, extra)


@pytest.fixture
def bond_features_extra(num_bonds, extra):
    return np.random.rand(num_bonds, extra)


@pytest.fixture
def mol_featurizer(atom_featurizer_mode, extra):
    atom_featurizer = get_multi_hot_atom_featurizer(atom_featurizer_mode)
    if atom_featurizer_mode == "RIGR":
        bond_featurizer = RIGRBondFeaturizer()
    else:
        bond_featurizer = MultiHotBondFeaturizer()
    return SimpleMoleculeMolGraphFeaturizer(atom_featurizer, bond_featurizer, extra, extra)


@pytest.fixture()
def batch_mol_featurizer(atom_featurizer_mode, extra):
    return CuikmolmakerMolGraphFeaturizer(
        atom_featurizer_mode, extra_atom_fdim=extra, extra_bond_fdim=extra, add_h=False
    )


@pytest.fixture
def bmg_simplemolecule(mols, mol_featurizer, atom_features_extra, bond_features_extra):
    n_atomss = [mol.GetNumAtoms() for mol in mols]
    n_bondss = [mol.GetNumBonds() for mol in mols]
    atom_features_extras = np.split(atom_features_extra, np.cumsum(n_atomss)[:-1])
    bond_features_extras = np.split(bond_features_extra, np.cumsum(n_bondss)[:-1])
    mgs = [
        mol_featurizer(mol, atom_features_extra, bond_features_extra)
        for mol, atom_features_extra, bond_features_extra in zip(
            mols, atom_features_extras, bond_features_extras
        )
    ]
    return BatchMolGraph(mgs)


@pytest.fixture
def bmg_cuikmolmaker(smis, batch_mol_featurizer, atom_features_extra, bond_features_extra):
    for smi in smis:
        print(smi)
    return batch_mol_featurizer(smis, atom_features_extra, bond_features_extra)


def test_V_shape(bmg_cuikmolmaker, num_atoms, batch_mol_featurizer, extra):
    bmg = bmg_cuikmolmaker
    assert bmg.V.shape[0] == num_atoms
    if batch_mol_featurizer.atom_featurizer_mode == "V1":
        atom_featurizer = MultiHotAtomFeaturizer.v1()
        assert bmg.V.shape[1] == 133 + extra
    elif batch_mol_featurizer.atom_featurizer_mode == "V2":
        atom_featurizer = MultiHotAtomFeaturizer.v2()
        assert bmg.V.shape[1] == 72 + extra
    elif batch_mol_featurizer.atom_featurizer_mode == "ORGANIC":
        atom_featurizer = MultiHotAtomFeaturizer.organic()
        assert bmg.V.shape[1] == 44 + extra
    elif batch_mol_featurizer.atom_featurizer_mode == "RIGR":
        atom_featurizer = RIGRAtomFeaturizer()
        assert bmg.V.shape[1] == 52 + extra
    assert batch_mol_featurizer.atom_fdim == len(atom_featurizer) + extra


def test_E_shape(bmg_cuikmolmaker, num_bonds, batch_mol_featurizer, extra):
    bmg = bmg_cuikmolmaker
    assert bmg.E.shape[0] == 2 * num_bonds
    if batch_mol_featurizer.atom_featurizer_mode == "RIGR":
        bond_featurizer = RIGRBondFeaturizer()
        assert bmg.E.shape[1] == 2 + extra
    else:
        bond_featurizer = MultiHotBondFeaturizer()
        assert bmg.E.shape[1] == 14 + extra
    assert batch_mol_featurizer.bond_fdim == len(bond_featurizer) + extra


def test_x2y_len(bmg_cuikmolmaker, num_bonds):
    assert bmg_cuikmolmaker.edge_index.shape == (2, 2 * num_bonds)
    assert bmg_cuikmolmaker.rev_edge_index.shape == (2 * num_bonds,)


def test_atom_extra_values(bmg_cuikmolmaker, atom_features_extra):
    if atom_features_extra.shape[1] == 0:
        return
    np.testing.assert_allclose(
        bmg_cuikmolmaker.V.numpy()[:, -atom_features_extra.shape[1] :], atom_features_extra
    )


def test_bond_extra(bmg_cuikmolmaker, extra, bond_features_extra):
    if bond_features_extra.shape[1] == 0:
        return
    np.testing.assert_allclose(
        bmg_cuikmolmaker.E.numpy()[::2, -bond_features_extra.shape[1] :], bond_features_extra
    )


def test_same_featurization(bmg_simplemolecule, bmg_cuikmolmaker):
    np.testing.assert_allclose(bmg_simplemolecule.V, bmg_cuikmolmaker.V.numpy())
    np.testing.assert_allclose(bmg_simplemolecule.E, bmg_cuikmolmaker.E.numpy())
    np.testing.assert_allclose(bmg_simplemolecule.edge_index, bmg_cuikmolmaker.edge_index.numpy())
    np.testing.assert_allclose(
        bmg_simplemolecule.rev_edge_index, bmg_cuikmolmaker.rev_edge_index.numpy()
    )
    np.testing.assert_allclose(bmg_simplemolecule.batch, bmg_cuikmolmaker.batch.numpy())
