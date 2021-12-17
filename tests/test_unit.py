"""Chemprop unit tests."""
import os
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory

from chemprop.data import get_header, preprocess_smiles_columns, get_task_names

class DataUtilsTests(TestCase):
    """
    Tests related to the loading and handling of data and features.
    """
    def test_get_header(self):
        # dummy data file
        with TemporaryDirectory() as temp_dir:
            dummy_path=os.path.join(temp_dir,'dummy_data.csv')
            with open(dummy_path,'w') as f:
                data = 'column0,column1\nCC,10\nCCC,15'
                f.write(data)
        
            # correct output
            header = get_header(dummy_path)
            self.assertEqual(header,['column0','column1'])

            # with path that doesn't exist
            bad_path=os.path.join(temp_dir,'bad_path.csv')
            with self.assertRaises(FileNotFoundError):
                get_header(bad_path)

    @patch(
        "chemprop.data.utils.get_header",
        lambda *args : [f'column{i}' for i in range(5)]
    )
    def test_preprocess_smiles_columns(self):
        # base case 1 molecule
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column0'])

        # 2 molecules
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=2,
        )
        self.assertEqual(smiles_columns, ['column0','column1'])

        # already specified columns
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=['column3'],
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

        # input not a list
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns='column3',
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

        # wrong number of molecules
        with self.assertRaises(ValueError):
            smiles_columns = preprocess_smiles_columns(
                path='dummy_path.txt',
                smiles_columns=['column3'],
                number_of_molecules=2,
            )

        # smiles columns not in file
        with self.assertRaises(ValueError):
            smiles_columns = preprocess_smiles_columns(
                path='dummy_path.txt',
                smiles_columns=['column3','not_in_file'],
                number_of_molecules=2,
            )

    @patch(
        "chemprop.data.utils.get_header",
        lambda *args : [f'column{i}' for i in range(5)]
    )
    @patch(
        "chemprop.data.utils.preprocess_smiles_columns",
        lambda *args, **kwargs : ['column0','column1'] # default smiles columns if unspecified
    )
    def test_get_task_names(self):
        # default behavior no arguments
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column2','column3','column4'])

        # behavior with smiles column specified
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=['column0'],
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1','column2','column3','column4'])

        # behavior with target columns specified
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=['column1', 'column3'],
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1', 'column3'])

        # behavior with ignore columns specified
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=['column2','column3'],
        )
        self.assertEqual(task_names, ['column4'])


class DatapointTests(TestCase):
    """
    Tests related to the MoleculeDatapoint and MoleculeDataset classes.
    """
    pass


class ModelTests(TestCase):
    """
    Tests related to the MoleculeModel class.
    """
    pass


class TrainTests(TestCase):
    """
    Tests related to training functions.
    """
    pass


class PredictTests(TestCase):
    """
    Tests related to predicting functions, including fingerprint.
    """


class HyperoptTests(TestCase):
    """
    Tests related to hyperparameter optimization
    """