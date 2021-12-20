"""Chemprop unit tests for chemprop/data/utils.py"""
import os
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory, TemporaryFile

from chemprop.data import get_header, preprocess_smiles_columns, get_task_names, get_data_weights , get_smiles\


class TestGetHeader(TestCase):
    """
    Tests of the get_header function.
    """
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        with open(os.path.join(self.temp_dir.name,'dummy_data.csv'),'w') as f:
            data = 'column0,column1\nCC,10\nCCC,15'
            f.write(data)
        
    def test_correct_file(self):
        """ Test correct input """
        header = get_header(os.path.join(self.temp_dir.name,'dummy_data.csv'))
        self.assertEqual(header,['column0','column1'])

    def test_bad_path(self):
        """Test bad provided path """
        bad_path=os.path.join(self.temp_dir.name,'bad_path.csv')
        with self.assertRaises(FileNotFoundError):
            get_header(bad_path)

    def tearDown(self):
        self.temp_dir.cleanup()


@patch(
    "chemprop.data.utils.get_header",
    lambda *args : [f'column{i}' for i in range(5)]
)
@patch(
    "os.path.isfile",
    lambda *args : True
)
class TestPreprocessSmiles(TestCase):
    """
    Tests of the preprocess_smiles_columns function
    """
    def test_basecase_1mol(self):
        """Test base case input with 1 molecule"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column0'])

    def test_2mol(self):
        """Test 2 molecule case"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=None,
            number_of_molecules=2,
        )
        self.assertEqual(smiles_columns, ['column0','column1'])

    def test_specified_smiles(self):
        """Test specified smiles column"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=['column3'],
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

    def test_out_of_order_smiles(self):
        """Test specified smiles columns provided in a different order"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns=['column3','column2'],
            number_of_molecules=2,
        )
        self.assertEqual(smiles_columns, ['column3','column2'])

    def test_input_not_list(self):
        """Test case where smiles_columns provided are not a list"""
        smiles_columns = preprocess_smiles_columns(
            path='dummy_path.txt',
            smiles_columns='column3',
            number_of_molecules=1,
        )
        self.assertEqual(smiles_columns, ['column3'])

    def test_wrong_num_mol(self):
        """Test that error is raised when wrong number of molecules provided"""
        with self.assertRaises(ValueError):
            smiles_columns = preprocess_smiles_columns(
                path='dummy_path.txt',
                smiles_columns=['column3'],
                number_of_molecules=2,
            )

    def test_smiles_not_in_file(self):
        """Test that error is raised whgen the provided smiles columns are not in the file"""
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
class TestGetTaskNames(TestCase):
    """
    Tests of the get_task_names function.
    """
    def test_default_no_arguments(self):
        """Testing default behavior no arguments"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column2','column3','column4'])

    def test_specified_smiles_columns(self):
        """Test behavior with smiles column specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=['column0'],
            target_columns=None,
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1','column2','column3','column4'])

    def test_specified_target_columns(self):
        """Test behavior with target columns specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=['column1', 'column3'],
            ignore_columns=None,
        )
        self.assertEqual(task_names, ['column1', 'column3'])

    def test_ignore_columns(self):
        """Test behavior with ignore columns specified"""
        task_names = get_task_names(
            path='dummy_path.txt',
            smiles_columns=None,
            target_columns=None,
            ignore_columns=['column2','column3'],
        )
        self.assertEqual(task_names, ['column4'])


class TestGetDataWeights(TestCase):
    """
    Tests for the function get_data_weights.
    """
    def setUp(self):
        self.temp_dir = TemporaryDirectory()

    def test_base_case(self):
        """Testing already normalized inputs"""
        path = os.path.join(self.temp_dir.name,'base_case.csv')
        with open(path, 'w') as f:
            f.write('weights\n1.5\n0.5\n1\n1\n1')
        weights = get_data_weights(path)
        self.assertEqual(weights,[1.5,0.5,1,1,1])

    def test_normalize_weights(self):
        """Testing weights that need to be normalized"""
        path = os.path.join(self.temp_dir.name,'normalize.csv')
        with open(path, 'w') as f:
            f.write('weights\n3\n1\n2\n2\n2')
        weights = get_data_weights(path)
        self.assertEqual(weights,[1.5,0.5,1,1,1])

    def test_negative_weights(self):
        """Testing behavior when one of the weights is negative"""
        path = os.path.join(self.temp_dir.name,'negativ.csv')
        with open(path, 'w') as f:
            f.write('weights\n3\n-1\n2\n2\n2')
        with self.assertRaises(ValueError):
            weights = get_data_weights(path)

    def tearDown(self):
        self.temp_dir.cleanup()


@patch(
    "chemprop.data.utils.preprocess_smiles_columns",
    lambda *args, **kwargs : ['column0','column1'] # default smiles columns if unspecified
)
class TestGetSmiles(TestCase):
    """
    Test for the get_smiles function.
    """
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.smiles_path = os.path.join(self.temp_dir.name,'smiles.csv')
        with open(self.smiles_path,'w') as f:
            f.write('column0,column1\nC,CC\nCC,CN\nO,CO')
        self.no_header_path = os.path.join(self.temp_dir.name,'no_header.csv')
        with open(self.no_header_path,'w') as f:
            f.write('C,CC\nCC,CN\nO,CO')

    def test_default_inputs(self):
        """Testing with no optional arguments."""
        smiles = get_smiles(
            path=self.smiles_path
        )
        self.assertEqual(smiles,[['C', 'CC'], ['CC', 'CN'], ['O', 'CO']])

    def test_specified_column_inputs(self):
        """Testing with a specified smiles column argument."""
        smiles = get_smiles(
            path=self.smiles_path,
            smiles_columns=['column1']
        )
        self.assertEqual(smiles,[['CC'], ['CN'], ['CO']])

    def test_specified_columns_changed_order(self):
        """Testing with no optional arguments."""
        smiles = get_smiles(
            path=self.smiles_path,
            smiles_columns=['column1','column0']
        )
        self.assertEqual(smiles,[['CC', 'C'], ['CN', 'CC'], ['CO', 'O']])

    def test_noheader_1mol(self):
        """Testing with no header"""
        smiles = get_smiles(
            path=self.no_header_path,
            header=False
        )
        self.assertEqual(smiles,[['C'], ['CC'], ['O']])

    def test_noheader_2mol(self):
        """Testing with no header and 2 molecules."""
        smiles = get_smiles(
            path=self.no_header_path,
            number_of_molecules=2,
            header=False
        )
        self.assertEqual(smiles,[['C', 'CC'], ['CC', 'CN'], ['O', 'CO']])

    def test_flatten(self):
        """Testing with flattened output"""
        smiles = get_smiles(
            path=self.smiles_path,
            flatten=True,
        )
        self.assertEqual(smiles,['C', 'CC', 'CC', 'CN', 'O', 'CO'])

    def tearDown(self):
        self.temp_dir.cleanup()

class TestFilterInvalidSmiles(TestCase):
    """
    Tests for the filter_invalid_smiles function.
    """
    pass