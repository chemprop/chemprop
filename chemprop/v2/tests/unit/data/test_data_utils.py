"""Chemprop unit tests for chemprop/data/utils.py"""
import numpy as np
import pytest
from rdkit import Chem
from astartes.utils.warnings import NormalizationWarning
from astartes.utils.exceptions import InvalidConfigurationError

from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.utils import split_data


@pytest.fixture
def molecule_dataset():
    """A dataset with single molecules"""
    smiles_list = ['C', 'CC', 'CCC', 'CN', 'CCN', 'CCCN', 'CCCCN', 'CO', 'CCO', 'CCCO']
    return [MoleculeDatapoint.from_smi(s) for s in smiles_list]

@pytest.fixture
def molecule_dataset_with_repeated_smiles():
    """A dataset with repeated single molecules"""
    smiles_list = ['C', 'CC', 'CN', 'CN', 'CO', 'C']
    return [MoleculeDatapoint.from_smi(s) for s in smiles_list]

def test_splits_sum1_warning(molecule_dataset):
    """Testing that the splits are normalized to 1"""
    with pytest.warns(NormalizationWarning):
        split_data(datapoints=molecule_dataset, sizes=(0.4, 0.6, 0.2))
        
def test_three_splits_provided(molecule_dataset):
    """Testing that three splits are provided"""
    with pytest.raises(AssertionError):
        split_data(datapoints=molecule_dataset, sizes=(0.8, 0.2))
        
def test_seed0(molecule_dataset):
    """Testing the random split with seed 0"""
    train, val, test = split_data(datapoints=molecule_dataset, seed=0)
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set(['CCO', 'CCN', 'CCCO', 'CC', 'CCCCN', 'CO', 'CN', 'C'])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set(['CCCN'])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set(['CCC'])

def test_seed100(molecule_dataset):
    """Testing the random split with seed 100"""
    train, val, test = split_data(datapoints=molecule_dataset, seed=100)
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set(['CCCCN', 'CC', 'CCCN', 'CCN', 'CCC', 'C', 'CN', 'CCCO'])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set(['CCO'])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set(['CO'])

def test_split_4_4_2(molecule_dataset):
    """Testing the random split with changed sizes"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0.4, 0.2))
    assert set([Chem.MolToSmiles(i.mol) for i in train]) == set(['CC', 'CCCO', 'CCN', 'CCO'])
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set(['C', 'CCCCN', 'CN', 'CO'])
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set(['CCC', 'CCCN'])
    
def test_split_4_0_6(molecule_dataset):
    """Testing the random split with an empty set"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0, 0.6))
    assert set([Chem.MolToSmiles(i.mol) for i in val]) == set([])

def test_repeated_smiles(molecule_dataset_with_repeated_smiles):
    """Testing the random split with repeated smiles"""
    train, val, test = split_data(datapoints=molecule_dataset_with_repeated_smiles, sizes=(0.8, 0.0, 0.2), split="random_with_repeated_smiles")
    assert [Chem.MolToSmiles(i.mol) for i in train] == ['CO', 'CC', 'C', 'C']
    assert [Chem.MolToSmiles(i.mol) for i in test] == ['CN', 'CN']

def test_kennard_stone(molecule_dataset):
    """Testing the Kennard-Stone split"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.4, 0.4, 0.2), split="kennard_stone")
    assert set([Chem.MolToSmiles(i.mol) for i in test]) == set(['CCCO', 'CCCN']) 
    
def test_kmeans(molecule_dataset):
    """Testing the KMeans split"""
    train, val, test = split_data(datapoints=molecule_dataset, sizes=(0.5, 0.0, 0.5), split="kmeans")
    assert [Chem.MolToSmiles(i.mol) for i in train] == ['CC', 'CC', 'CC', 'CC', 'CC', 'CC', 'CC']
    assert [Chem.MolToSmiles(i.mol) for i in test] == ['C', 'C', 'C']