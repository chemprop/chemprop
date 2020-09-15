"""Chemprop integration tests."""
import json
import os
from tempfile import TemporaryDirectory
from typing import List
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
from parameterized import parameterized

from chemprop.constants import TEST_SCORES_FILE_NAME
from chemprop.train import chemprop_train, chemprop_predict
from chemprop.hyperparameter_optimization import chemprop_hyperopt


TEST_DATA_DIR = 'tests/data'
SEED = 0
EPOCHS = 10
NUM_FOLDS = 3
NUM_ITER = 2


class ChempropTests(TestCase):
    @staticmethod
    def create_raw_train_args(dataset_type: str,
                              metric: str,
                              save_dir: str,
                              flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for training."""
        return [
            'chemprop_train',  # Note: not actually used, just a placeholder
            '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}.csv'),
            '--dataset_type', dataset_type,
            '--epochs', str(EPOCHS),
            '--num_folds', str(NUM_FOLDS),
            '--seed', str(SEED),
            '--metric', metric,
            '--save_dir', save_dir,
            '--quiet'
        ] + (flags if flags is not None else [])

    @staticmethod
    def create_raw_predict_args(dataset_type: str,
                                preds_path: str,
                                checkpoint_dir: str,
                                flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for predicting."""
        return [
            'chemprop_predict',  # Note: not actually used, just a placeholder
            '--test_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_smiles.csv'),
            '--preds_path', preds_path,
            '--checkpoint_dir', checkpoint_dir
        ] + (flags if flags is not None else [])

    @staticmethod
    def create_raw_hyperopt_args(dataset_type: str,
                                 config_save_path: str,
                                 save_dir: str,
                                 flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for hyperparameter optimization."""
        return [
            'chemprop_hyperopt',  # Note: not actually used, just a placeholder
            '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}.csv'),
            '--dataset_type', dataset_type,
            '--epochs', str(EPOCHS),
            '--num_iter', str(NUM_ITER),
            '--config_save_path', config_save_path,
            '--save_dir', save_dir,
            '--quiet'
        ] + (flags if flags is not None else [])

    def train(self,
              dataset_type: str,
              metric: str,
              save_dir: str,
              flags: List[str] = None):
        # Set up command line arguments for training
        raw_train_args = self.create_raw_train_args(
            dataset_type=dataset_type,
            metric=metric,
            save_dir=save_dir,
            flags=flags
        )

        # Train
        with patch('sys.argv', raw_train_args):
            chemprop_train()

    def predict(self,
                dataset_type: str,
                preds_path: str,
                save_dir: str,
                flags: List[str] = None):
        # Set up command line arguments for predicting
        raw_predict_args = self.create_raw_predict_args(
            dataset_type=dataset_type,
            preds_path=preds_path,
            checkpoint_dir=save_dir,
            flags=flags
        )

        # Predict
        with patch('sys.argv', raw_predict_args):
            chemprop_predict()

    def hyperopt(self,
                 dataset_type: str,
                 config_save_path: str,
                 save_dir: str,
                 flags: List[str] = None):
        # Set up command line arguments for training
        raw_hyperopt_args = self.create_raw_hyperopt_args(
            dataset_type=dataset_type,
            config_save_path=config_save_path,
            save_dir=save_dir,
            flags=flags
        )

        # Predict
        with patch('sys.argv', raw_hyperopt_args):
            chemprop_hyperopt()

    @parameterized.expand([
        ('default', [], 1.237620),
        ('morgan_features_generator', ['--features_generator', 'morgan'], 1.834947),
        ('rdkit_features_path', ['--features_path', os.path.join(TEST_DATA_DIR, 'regression.npz'), '--no_features_scaling'], 0.807828)
    ])
    def test_chemprop_train_single_task_regression(self,
                                                   name: str,
                                                   train_flags: List[str],
                                                   expected_score: float):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'rmse'
            self.train(dataset_type='regression', metric=metric, save_dir=save_dir, flags=train_flags)

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 1)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, expected_score, delta=0.02)

    @parameterized.expand([
        ('default', [], 0.691205),
        ('morgan_features_generator', ['--features_generator', 'morgan'], 0.619021),
        ('rdkit_features_path', ['--features_path', os.path.join(TEST_DATA_DIR, 'classification.npz'), '--no_features_scaling'], 0.659145)
    ])
    def test_chemprop_train_multi_task_classification(self,
                                                      name: str,
                                                      train_flags: List[str],
                                                      expected_score: float):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'auc'
            self.train(dataset_type='classification', metric=metric, save_dir=save_dir, flags=train_flags)

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 12)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, expected_score, delta=0.02)

    @parameterized.expand([
        ('default', [], [], 0.561477),
        ('morgan_features_generator', ['--features_generator', 'morgan'], ['--features_generator', 'morgan'], 3.905965),
        ('rdkit_features_path',
         ['--features_path', os.path.join(TEST_DATA_DIR, 'regression.npz'), '--no_features_scaling'],
         ['--features_path', os.path.join(TEST_DATA_DIR, 'regression_test.npz'), '--no_features_scaling'],
         0.693359)
    ])
    def test_chemprop_predict_single_task_regression(self,
                                                     name: str,
                                                     train_flags: List[str],
                                                     predict_flags: List[str],
                                                     expected_score: float):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'regression'
            self.train(dataset_type=dataset_type, metric='rmse', save_dir=save_dir, flags=train_flags)

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(dataset_type=dataset_type, preds_path=preds_path, save_dir=save_dir, flags=predict_flags)

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.nanmean((pred - true) ** 2))
            self.assertAlmostEqual(mse, expected_score, delta=0.02)

    @parameterized.expand([
        ('default', [], [], 0.064605),
        ('morgan_features_generator', ['--features_generator', 'morgan'], ['--features_generator', 'morgan'], 0.083170),
        ('rdkit_features_path',
         ['--features_path', os.path.join(TEST_DATA_DIR, 'classification.npz'), '--no_features_scaling'],
         ['--features_path', os.path.join(TEST_DATA_DIR, 'classification_test.npz'), '--no_features_scaling'],
         0.064972)
    ])
    def test_chemprop_predict_multi_task_classification(self,
                                                        name: str,
                                                        train_flags: List[str],
                                                        predict_flags: List[str],
                                                        expected_score: float):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'classification'
            self.train(dataset_type=dataset_type, metric='auc', save_dir=save_dir, flags=train_flags)

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(dataset_type=dataset_type, preds_path=preds_path, save_dir=save_dir, flags=predict_flags)

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.nanmean((pred - true) ** 2))
            self.assertAlmostEqual(mse, expected_score, delta=0.02)

    def test_chemprop_hyperopt(self):
        with TemporaryDirectory() as save_dir:
            # Train
            config_save_path = os.path.join(save_dir, 'config.json')
            self.hyperopt(dataset_type='regression', config_save_path=config_save_path, save_dir=save_dir)

            # Check results
            with open(config_save_path) as f:
                config = json.load(f)

            parameters = {'depth': (2, 6), 'hidden_size': (300, 2400), 'ffn_num_layers': (1, 3), 'dropout': (0.0, 0.4)}

            self.assertEqual(set(config.keys()), set(parameters.keys()))

            for parameter, (min_value, max_value) in parameters.items():
                self.assertTrue(min_value <= config[parameter] <= max_value)


if __name__ == '__main__':
    unittest.main()
