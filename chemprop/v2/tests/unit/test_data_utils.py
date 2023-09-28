"""Chemprop unit tests for chemprop/data/utils.py"""
from unittest import TestCase

from chemprop.v2.data.datasets import MoleculeDataset
from chemprop.v2.data.datapoints import MoleculeDatapoint
from chemprop.v2.data.utils import split_data


class TestSplitData(TestCase):
    """
    Testing of the split_data function.
    """

    # Testing currently covers random and random_with_repeated_smiles
    def setUp(self):
        smiles_list = [
            ["C", "CC"],
            ["CC", "CCC"],
            ["CCC", "CN"],
            ["CN", "CCN"],
            ["CCN", "CCCN"],
            ["CCCN", "CCCCN"],
            ["CCCCN", "CO"],
            ["CO", "CCO"],
            ["CO", "CCCO"],
            ["CN", "CCC"],
        ]
        self.dataset = MoleculeDataset([MoleculeDatapoint(s) for s in smiles_list])

    def test_splits_sum1(self):
        with self.assertRaises(ValueError):
            train, val, test = split_data(data=self.dataset, sizes=(0.4, 0.8, 0.2))

    def test_three_splits_provided(self):
        with self.assertRaises(ValueError):
            train, val, test = split_data(data=self.dataset, sizes=(0.8, 0.2))

    def test_random_split(self):
        """Testing the random split with seed 0"""
        train, val, test = split_data(data=self.dataset)
        self.assertEqual(
            train.smiles(),
            [
                ["CO", "CCO"],
                ["CO", "CCCO"],
                ["CC", "CCC"],
                ["CCCN", "CCCCN"],
                ["CN", "CCN"],
                ["CCN", "CCCN"],
                ["CCC", "CN"],
                ["C", "CC"],
            ],
        )

    def test_seed1(self):
        """Testing the random split with seed 1"""
        train, val, test = split_data(data=self.dataset, seed=1)
        self.assertEqual(
            train.smiles(),
            [
                ["CCCCN", "CO"],
                ["CO", "CCCO"],
                ["CN", "CCC"],
                ["CO", "CCO"],
                ["CCCN", "CCCCN"],
                ["CN", "CCN"],
                ["C", "CC"],
                ["CCN", "CCCN"],
            ],
        )

    def test_split_4_4_2(self):
        """Testing the random split with changed sizes"""
        train, val, test = split_data(data=self.dataset, sizes=(0.4, 0.4, 0.2))
        self.assertEqual(
            train.smiles(), [["CO", "CCO"], ["CO", "CCCO"], ["CC", "CCC"], ["CCCN", "CCCCN"]]
        )

    def test_split_4_0_6(self):
        """Testing the random split with an empty set"""
        train, val, test = split_data(data=self.dataset, sizes=(0.4, 0, 0.6))
        self.assertEqual(val.smiles(), [])

    def test_repeated_smiles(self):
        """Testing the random split with repeated smiles"""
        train, val, test = split_data(
            data=self.dataset, sizes=(0.4, 0.4, 0.2), split="random_with_repeated_smiles"
        )
        self.assertEqual(test.smiles(), [["CO", "CCCO"], ["CO", "CCO"]])

    def test_kmeans(self):
        """Testing the KMeans split"""
        train, val, test = split_data(data=self.dataset, sizes=(0.5, 0.0, 0.5), split="kmeans")
        self.assertEqual(test.smiles(), [["C", "CC"], ["C", "CC"], ["C", "CC"]])

    def test_kennard_stone(self):
        """Testing the Kennard-Stone split"""
        train, val, test = split_data(
            data=self.dataset, sizes=(0.4, 0.4, 0.2), split="kennard_stone"
        )
        self.assertEqual(test.smiles(), [["CO", "CCCO"], ["CN", "CCC"]])
