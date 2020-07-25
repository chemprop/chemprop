"""Chemprop integration tests."""
import os
from tempfile import TemporaryDirectory
from typing import List
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from chemprop.constants import TEST_SCORES_FILE_NAME
from chemprop.train import chemprop_train, chemprop_predict


TEST_DATA_DIR = 'tests/data'
SEED = 0
EPOCHS = 10
NUM_FOLDS = 3


class ChempropTests(TestCase):
    @staticmethod
    def create_raw_train_args(dataset_type: str, metric: str, save_dir: str) -> List[str]:
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
        ]

    @staticmethod
    def create_raw_predict_args(dataset_type: str, preds_path: str, checkpoint_dir: str) -> List[str]:
        """Creates a list of raw command line arguments for predicting."""
        return [
            'chemprop_predict',  # Note: not actually used, just a placeholder
            '--test_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_smiles.csv'),
            '--preds_path', preds_path,
            '--checkpoint_dir', checkpoint_dir
        ]

    def train(self, dataset_type: str, metric: str, save_dir: str):
        # Set up command line arguments for training
        raw_train_args = self.create_raw_train_args(
            dataset_type=dataset_type,
            metric=metric,
            save_dir=save_dir
        )

        # Train
        with patch('sys.argv', raw_train_args):
            chemprop_train()

    def predict(self, dataset_type: str, preds_path: str, save_dir: str):
        # Set up command line arguments for predicting
        raw_predict_args = self.create_raw_predict_args(
            dataset_type=dataset_type,
            preds_path=preds_path,
            checkpoint_dir=save_dir
        )

        # Predict
        with patch('sys.argv', raw_predict_args):
            chemprop_predict()

    def test_chemprop_train_single_task_regression(self):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'rmse'
            self.train(dataset_type='regression', metric=metric, save_dir=save_dir)

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 1)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, 1.237620, delta=0.02)

    def test_chemprop_train_multi_task_classification(self):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'auc'
            self.train(dataset_type='classification', metric=metric, save_dir=save_dir)

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 12)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, 0.679375, delta=0.02)

    def test_chemprop_predict_single_task_regression(self):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'regression'
            self.train(dataset_type=dataset_type, metric='rmse', save_dir=save_dir)

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(dataset_type=dataset_type, preds_path=preds_path, save_dir=save_dir)

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.nanmean((pred - true) ** 2))
            self.assertAlmostEqual(mse, 0.559111, delta=0.02)

    def test_chemprop_predict_multi_task_classification(self):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'classification'
            self.train(dataset_type=dataset_type, metric='auc', save_dir=save_dir)

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(dataset_type=dataset_type, preds_path=preds_path, save_dir=save_dir)

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.nanmean((pred - true) ** 2))
            self.assertAlmostEqual(mse, 0.064600, delta=0.02)


if __name__ == '__main__':
    unittest.main()
