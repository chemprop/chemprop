"""Chemprop integration tests."""
import os
from tempfile import TemporaryDirectory
from typing import List
import unittest
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from sklearn.metrics import mean_squared_error

from chemprop.constants import TEST_SCORES_FILE_NAME
from chemprop.train import chemprop_train, chemprop_predict


SEED = 0
EPOCHS = 10
NUM_FOLDS = 3


class ChempropTests(TestCase):
    @staticmethod
    def create_raw_train_args(dataset_type: str, metric: str, save_dir: str) -> List[str]:
        """Creates a list of raw command line arguments for training."""
        return [
            'chemprop_predict',
            '--data_path', f'data/{dataset_type}.csv',
            '--dataset_type', dataset_type,
            '--epochs', str(EPOCHS),
            '--num_folds', str(NUM_FOLDS),
            '--seed', str(SEED),
            '--metric', metric,
            '--save_dir', save_dir,
            '--quiet'
        ]

    @staticmethod
    def create_raw_predict_args(preds_path: str, checkpoint_dir: str) -> List[str]:
        """Creates a list of raw command line arguments for predicting."""
        return [
            'chemprop_train',
            '--test_path', f'data/test_smiles.csv',
            '--preds_path', preds_path,
            '--checkpoint_dir', checkpoint_dir
        ]

    def test_chemprop_train_single_task_regression(self):
        with TemporaryDirectory() as save_dir:
            # Set up command line arguments for training
            dataset_type = 'regression'
            metric = 'rmse'
            raw_train_args = self.create_raw_train_args(
                dataset_type=dataset_type,
                metric=metric,
                save_dir=save_dir
            )

            # Train
            with patch('sys.argv', raw_train_args):
                chemprop_train()

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 1)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, 1.500242, delta=0.1)

    def test_chemprop_train_multi_task_classification(self):
        with TemporaryDirectory() as save_dir:
            # Set up command line arguments for training
            dataset_type = 'classification'
            metric = 'auc'
            raw_train_args = self.create_raw_train_args(
                dataset_type=dataset_type,
                metric=metric,
                save_dir=save_dir
            )

            # Train
            with patch('sys.argv', raw_train_args):
                chemprop_train()

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = test_scores_data[f'Mean {metric}']
            self.assertEqual(len(test_scores), 12)

            mean_score = test_scores.mean()
            self.assertAlmostEqual(mean_score, 0.679375, delta=0.1)

    def test_chemprop_predict_multi_task_classification(self):
        with TemporaryDirectory() as save_dir:
            # Set up command line arguments for training
            dataset_type = 'classification'
            metric = 'auc'
            raw_train_args = self.create_raw_train_args(
                dataset_type=dataset_type,
                metric=metric,
                save_dir=save_dir
            )

            # Train
            with patch('sys.argv', raw_train_args):
                chemprop_train()

            # Set up command line arguments for predicting
            preds_path = os.path.join(save_dir, 'preds.csv')
            raw_predict_args = self.create_raw_predict_args(
                preds_path=preds_path,
                checkpoint_dir=save_dir
            )

            # Predict
            with patch('sys.argv', raw_predict_args):
                chemprop_predict()

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv('data/test_true.csv')
            self.assertEqual(pred.keys(), true.keys())
            self.assertEqual(pred['smiles'], true['smiles'])

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = mean_squared_error(true, pred)
            print(f'MSE  = {mse}')
            self.assertAlmostEqual(mse, 0.1, delta=0.1)


class SklearnTests(TestCase):
    def test_rf_train_single_task_regression(self):
        pass

    def test_svm_train_single_task_regression(self):
        pass


if __name__ == '__main__':
    unittest.main()
