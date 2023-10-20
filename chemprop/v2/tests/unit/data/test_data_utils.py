"""Chemprop unit tests for chemprop/data/utils.py"""
import pytest
import numpy as np
from astartes import train_val_test_split
from astartes.utils.warnings import NormalizationWarning
from rdkit import Chem

from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.utils import split_data, _unpack_astartes_result


@pytest.fixture
def molecule_dataset():
    """A dataset with single molecules"""
    smiles_list = ["C", "CC", "CCC", "CN", "CCN", "CCCN", "CCCCN", "CO", "CCO", "CCCO"]
    return [MoleculeDatapoint.from_smi(s) for s in smiles_list]


@pytest.fixture
def molecule_dataset_with_repeated_smiles():
    """A dataset with repeated single molecules"""
    smiles_list = ["C", "CC", "CN", "CN", "CO", "C"]
    return [MoleculeDatapoint.from_smi(s) for s in smiles_list]


@pytest.fixture
def molecule_dataset_with_rings():
    """A dataset with rings (for scaffold splitting)"""
    smiles_list = ["C", "CC", "CCC", "C1CC1", "C1CCC1"]
    return [MoleculeDatapoint.from_smi(s) for s in smiles_list]


def test_splits_sum1_warning(molecule_dataset):
    """Testing that the splits are normalized to 1, for overspecified case."""
    with pytest.warns(NormalizationWarning):
        split_data(datapoints=molecule_dataset, sizes=(0.4, 0.6, 0.2))
        
def test_splits_sum2_warning(molecule_dataset):
    """Testing that the splits are normalized to 1, for underspecified case."""
    with pytest.warns(NormalizationWarning):
        split_data(datapoints=molecule_dataset, sizes=(0.1, 0.1, 0.1))
        
        
def test_three_splits_provided(molecule_dataset):
    """Testing that three splits are provided"""
    with pytest.raises(AssertionError):
        split_data(datapoints=molecule_dataset, sizes=(0.8, 0.2))


def test_seed0(molecule_dataset):
    """
    Testing that split_data can get expected output using astartes as backend for random split with seed 0.
    Note: the behaviour of randomness for data splitting is not controlled by Chemprop but by the chosen backend.
    """ 
    train, val, test = split_data(datapoints=molecule_dataset, seed=0)
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(molecule_dataset,
                                                                          train_val_test_split(np.arange(len(molecule_dataset)), sampler="random", random_state=0),
                                                                          True)
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set([Chem.MolToSmiles(i.mol) for i in train_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set([Chem.MolToSmiles(i.mol) for i in val_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set([Chem.MolToSmiles(i.mol) for i in test_astartes])
    
def test_seed100(molecule_dataset):
    """
    Testing that split_data can get expected output using astartes as backend for random split with seed 100.
    Note: the behaviour of randomness for data splitting is not controlled by Chemprop but by the chosen backend.
    """ 
    train, val, test = split_data(datapoints=molecule_dataset, seed=100)
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(molecule_dataset,
                                                                          train_val_test_split(np.arange(len(molecule_dataset)), sampler="random", random_state=100),
                                                                          True)
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set([Chem.MolToSmiles(i.mol) for i in train_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set([Chem.MolToSmiles(i.mol) for i in val_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set([Chem.MolToSmiles(i.mol) for i in test_astartes])


def test_split_4_4_2(molecule_dataset):
    """Testing the random split with changed sizes"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0.4, 0.2))
    train_astartes, val_astartes, test_astartes = _unpack_astartes_result(molecule_dataset,
                                                                        train_val_test_split(np.arange(len(molecule_dataset)), sampler="random", train_size=0.4, val_size=0.4, test_size=0.2, random_state=0),
                                                                        True)
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set([Chem.MolToSmiles(i.mol) for i in train_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set([Chem.MolToSmiles(i.mol) for i in val_astartes])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set([Chem.MolToSmiles(i.mol) for i in test_astartes])


def test_split_empty_validation_set(molecule_dataset):
    """Testing the random split with an empty validation set"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0, 0.6))
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set([])


def test_random_split(molecule_dataset_with_repeated_smiles):
    """
    Testing if random split yield expected results.
    Note: This test mainly serves as a red flag. Test faliure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "random"
    train, val, test = split_data(datapoints=molecule_dataset_with_repeated_smiles, sizes=(0.4, 0.4, 0.2), split=split_type)
    
    fail_msg = f"Data splitting with type {split_type} has failed. The failure is likely due to a change of the corresponding data splitting backend. Chemprop itself does not handle data splitting. The test here mainly serve as a red flag."
    assert [Chem.MolToSmiles(i.mol) for i in train] == ['CN', 'CC'], fail_msg

def test_repeated_smiles(molecule_dataset_with_repeated_smiles):
    """
    Testing if random split with repeated smiles yield expected results.
    Note: This test mainly serves as a red flag. Test faliure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "random_with_repeated_smiles"
    train, val, test = split_data(datapoints=molecule_dataset_with_repeated_smiles, sizes=(0.8, 0.0, 0.2), split=split_type)
    
    fail_msg = f"Data splitting with type {split_type} has failed. The failure is likely due to a change of the corresponding data splitting backend. Chemprop itself does not handle data splitting. The test here mainly serve as a red flag."
    assert [Chem.MolToSmiles(i.mol) for i in train] == ["CO", "CC", "C", "C"], fail_msg
    assert [Chem.MolToSmiles(i.mol) for i in test] == ["CN", "CN"], fail_msg


def test_kennard_stone(molecule_dataset):
    """
    Testing if Kennard-Stone split yield expected results.
    Note: This test mainly serves as a red flag. Test faliure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "kennard_stone"
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0.4, 0.2), split=split_type)
    
    fail_msg = f"Data splitting with type {split_type} has failed. The failure is likely due to a change of the corresponding data splitting backend. Chemprop itself does not handle data splitting. The test here mainly serve as a red flag."
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set(["CCCO", "CCCN"]), fail_msg


def test_kmeans(molecule_dataset):
    """
    Testing if Kmeans split yield expected results.
    Note: This test mainly serves as a red flag. Test faliure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "kmeans"
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.5, 0.0, 0.5), split=split_type)
    
    fail_msg = f"Data splitting with type {split_type} has failed. The failure is likely due to a change of the corresponding data splitting backend. Chemprop itself does not handle data splitting. The test here mainly serve as a red flag."
    assert [Chem.MolToSmiles(i.mol) for i in train] == ["C", "CC", "CCC", "CN", "CO", "CCO", "CCCO"], fail_msg


def test_scaffold(molecule_dataset_with_rings):
    """
    Testing if Bemis-Murcko Scaffolds split yield expected results.
    Note: This test mainly serves as a red flag. Test faliure strongly indicates unexpected change of data splitting backend that needs attention.
    """
    split_type = "scaffold_balanced"
    train, val, test = split_data(datapoints=molecule_dataset_with_rings, sizes=(0.3, 0.3, 0.3), split=split_type)
    
    fail_msg = f"Data splitting with type {split_type} has failed. The failure is likely due to a change of the corresponding data splitting backend. Chemprop itself does not handle data splitting. The test here mainly serve as a red flag."
    assert [Chem.MolToSmiles(i.mol) for i in train] == ['C', 'CC', 'CCC'], fail_msg