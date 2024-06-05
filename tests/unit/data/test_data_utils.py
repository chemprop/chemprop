from astartes import train_val_test_split
from astartes.utils.warnings import NormalizationWarning
import numpy as np
import pytest
from rdkit import Chem

from chemprop.data.splitting import _unpack_astartes_result, make_split_indices


@pytest.fixture(params=[["C", "CC", "CCC", "CN", "CCN", "CCCN", "CCCCN", "CO", "CCO", "CCCO"]])
def mol_data(request):
    """A dataset with single molecules"""
    return [Chem.MolFromSmiles(smi) for smi in request.param]


@pytest.fixture(params=[["C", "CC", "CN", "CN", "CO", "C"]])
def mol_data_with_repeated_mols(request):
    """A dataset with repeated single molecules"""
    return [Chem.MolFromSmiles(smi) for smi in request.param]


@pytest.fixture(params=[["C", "CC", "CCC", "C1CC1", "C1CCC1"]])
def molecule_dataset_with_rings(request):
    """A dataset with rings (for scaffold splitting)"""
    return [Chem.MolFromSmiles(smi) for smi in request.param]


def test_splits_sum1_warning(mol_data):
    """Testing that the splits are normalized to 1, for overspecified case."""
    with pytest.warns(NormalizationWarning):
        make_split_indices(mols=mol_data, sizes=(0.4, 0.6, 0.2))


def test_splits_sum2_warning(mol_data):
    """Testing that the splits are normalized to 1, for underspecified case."""
    with pytest.warns(NormalizationWarning):
        make_split_indices(mols=mol_data, sizes=(0.1, 0.1, 0.1))


def test_three_splits_provided(mol_data):
    """Testing that three splits are provided"""
    with pytest.raises(ValueError):
        make_split_indices(mols=mol_data, sizes=(0.8, 0.2))


def test_seed0(mol_data):
    """
    Testing that make_split_indices can get expected output using astartes as backend for random split with seed 0.
    Note: the behaviour of randomness for data splitting is not controlled by chemprop but by the chosen backend.
    """
    train, val, test = make_split_indices(mols=mol_data, seed=0)
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(
        train_val_test_split(np.arange(len(mol_data)), sampler="random", random_state=0), True
    )
    assert set(train) == set(train_astartes)
    assert set(val) == set(val_astartes)
    assert set(test) == set(test_astartes)


def test_seed100(mol_data):
    """
    Testing that make_split_indices can get expected output using astartes as backend for random split with seed 100.
    Note: the behaviour of randomness for data splitting is not controlled by chemprop but by the chosen backend.
    """
    train, val, test = make_split_indices(mols=mol_data, seed=100)
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(
        train_val_test_split(np.arange(len(mol_data)), sampler="random", random_state=100), True
    )
    assert set(train) == set(train_astartes)
    assert set(val) == set(val_astartes)
    assert set(test) == set(test_astartes)


def test_split_4_4_2(mol_data):
    """Testing the random split with changed sizes"""
    train, val, test = make_split_indices(mols=mol_data, sizes=(0.4, 0.4, 0.2))
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(
        train_val_test_split(
            np.arange(len(mol_data)),
            sampler="random",
            train_size=0.4,
            val_size=0.4,
            test_size=0.2,
            random_state=0,
        ),
        True,
    )
    assert set(train) == set(train_astartes)
    assert set(val) == set(val_astartes)
    assert set(test) == set(test_astartes)


def test_split_empty_validation_set(mol_data):
    """Testing the random split with an empty validation set"""
    train, val, test = make_split_indices(mols=mol_data, sizes=(0.4, 0, 0.6))
    assert set(val) == set([])


def test_random_split(mol_data_with_repeated_mols):
    """
    Testing if random split yield expected results.
    Note: This test mainly serves as a red flag. Test failure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "random"
    train, val, test = make_split_indices(
        mols=mol_data_with_repeated_mols, sizes=(0.4, 0.4, 0.2), split=split_type
    )

    assert train == [2, 1]


def test_repeated_smiles(mol_data_with_repeated_mols):
    """
    Testing if random split with repeated smiles yield expected results.
    Note: This test mainly serves as a red flag. Test failure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "random_with_repeated_smiles"
    train, val, test = make_split_indices(
        mols=mol_data_with_repeated_mols, sizes=(0.8, 0.0, 0.2), split=split_type
    )

    assert train == [4, 1, 0, 5]
    assert test == [2, 3]


def test_kennard_stone(mol_data):
    """
    Testing if Kennard-Stone split yield expected results.
    Note: This test mainly serves as a red flag. Test failure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "kennard_stone"
    train, val, test = make_split_indices(mols=mol_data, sizes=(0.4, 0.4, 0.2), split=split_type)

    assert set(test) == set([9, 5])


def test_kmeans(mol_data):
    """
    Testing if Kmeans split yield expected results.
    Note: This test mainly serves as a red flag. Test failure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "kmeans"
    train, val, test = make_split_indices(mols=mol_data, sizes=(0.5, 0.0, 0.5), split=split_type)

    assert train == [0, 1, 2, 3, 7, 8, 9]


def test_scaffold(molecule_dataset_with_rings):
    """
    Testing if Bemis-Murcko Scaffolds split yield expected results.
    Note: This test mainly serves as a red flag. Test failure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "scaffold_balanced"
    train, val, test = make_split_indices(
        mols=molecule_dataset_with_rings, sizes=(0.3, 0.3, 0.3), split=split_type
    )

    assert train == [0, 1, 2]
