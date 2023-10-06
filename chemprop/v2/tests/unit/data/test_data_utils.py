"""Chemprop unit tests for chemprop/data/utils.py"""
import numpy as np
import pytest
from rdkit import Chem

from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.utils import split_data


class TestSplitData:
    """
    Testing of the split_data function.
    """

    def setup_class(self):
        smiles_list = [
            "C",
            "CC",
            "CCC",
            "CN",
            "CCN",
            "CCCN",
            "CCCCN",
            "CO",
            "CO",
            "CN"
        ]
        self.dataset = [MoleculeDatapoint.from_smi(s) for s in smiles_list]

    def test_splits_sum1(self):
        with pytest.raises(ValueError):
            train, val, test = split_data(datapoints=self.dataset, sizes=(0.4, 0.8, 0.2))

    def test_three_splits_provided(self):
        with pytest.raises(ValueError):
            train, val, test = split_data(datapoints=self.dataset, sizes=(0.8, 0.2))

    def test_random_split(self):
        """Testing the random split with seed 0"""
        train, val, test = split_data(datapoints=self.dataset)
        assert np.testing.assert_array_equal(
            [Chem.MolToSmiles(i.mol) for i in train],
            ['CO', 'CCN', 'CN', 'CC', 'CCCCN', 'CO', 'CN', 'C'],
        ) is None

    def test_seed1(self):
        """Testing the random split with seed 1"""
        train, val, test = split_data(datapoints=self.dataset, seed=1)
        assert np.testing.assert_array_equal(
            [Chem.MolToSmiles(i.mol) for i in train],
            ['CN', 'CCCCN', 'CCN', 'C', 'CN', 'CC', 'CO', 'CO'],
        ) is None

    def test_split_4_4_2(self):
        """Testing the random split with changed sizes"""
        train, val, test = split_data(datapoints=self.dataset, sizes=(0.4, 0.4, 0.2))
        assert np.testing.assert_array_equal([Chem.MolToSmiles(i.mol) for i in train], ['CO', 'CCN', 'CN', 'CC']) is None

    def test_split_4_0_6(self):
        """Testing the random split with an empty set"""
        train, val, test = split_data(datapoints=self.dataset, sizes=(0.4, 0, 0.6))
        assert np.testing.assert_array_equal(val, []) is None

    def test_repeated_smiles(self):
        """Testing the random split with repeated smiles"""
        train, val, test = split_data(datapoints=self.dataset, sizes=(0.4, 0.4, 0.2), split="random_with_repeated_smiles")
        assert np.testing.assert_array_equal([Chem.MolToSmiles(i.mol) for i in test], ["CO", "CO"]) is None

    def test_kennard_stone(self):
        """Testing the Kennard-Stone split"""
        train, val, test = split_data(datapoints=self.dataset, sizes=(0.4, 0.4, 0.2), split="kennard_stone")
        assert np.testing.assert_array_equal([Chem.MolToSmiles(i.mol) for i in test], ["CO", "CN"]) is None

    def test_kmeans(self):
        """Testing the KMeans split"""
        train, val, test = split_data(datapoints=self.dataset, sizes=(0.5, 0.0, 0.5), split="kmeans")
        assert np.testing.assert_array_equal([Chem.MolToSmiles(i.mol) for i in test], ["C", "C", "C"]) is None
