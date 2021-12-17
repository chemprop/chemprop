"""Chemprop unit tests."""
from unittest import TestCase
from unittest.mock import patch

from chemprop.data import get_task_names

class DataUtilsTests(TestCase):
    """
    Tests related to the loading and handling of data and features.
    """
    def test_preprocess_smiles_columns(self):
        pass

    @patch(
        "chemprop.data.utils.get_header",
        lambda x : [f'column{i}' for i in range(5)]
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