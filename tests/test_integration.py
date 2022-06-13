"""Chemprop integration tests."""
from flask import url_for
from io import BytesIO
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
from chemprop.hyperparameter_optimization import chemprop_hyperopt
from chemprop.interpret import chemprop_interpret
from chemprop.sklearn_predict import sklearn_predict
from chemprop.sklearn_train import sklearn_train
from chemprop.train import chemprop_train, chemprop_predict, evaluate_predictions, chemprop_fingerprint
from chemprop.web.wsgi import build_app
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.features import load_features


TEST_DATA_DIR = 'tests/data'
SEED = 0
EPOCHS = 3
NUM_FOLDS = 3
NUM_ITER = 2
DELTA = 0.015


class ChempropTests(TestCase):
    @staticmethod
    def create_raw_train_args(dataset_type: str,
                              metric: str,
                              save_dir: str,
                              model_type: str = 'chemprop',
                              flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for training."""
        return [
            'train',  # Note: not actually used, just a placeholder
            '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}.csv'), # Note: adding another --data_path argument will overwrite this one
            '--dataset_type', dataset_type,
            '--epochs', str(EPOCHS),
            '--num_folds', str(NUM_FOLDS),
            '--seed', str(SEED),
            '--metric', metric,
            '--save_dir', save_dir,
            '--quiet',
            '--empty_cache'
        ] + (['--model_type', model_type] if model_type != 'chemprop' else []) + (flags if flags is not None else [])

    @staticmethod
    def create_raw_predict_args(dataset_type: str,
                                preds_path: str,
                                checkpoint_dir: str,
                                flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for predicting."""
        return [
            'predict',  # Note: not actually used, just a placeholder
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
            'hyperopt',  # Note: not actually used, just a placeholder
            '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}.csv'),
            '--dataset_type', dataset_type,
            '--epochs', str(EPOCHS),
            '--num_iter', str(NUM_ITER),
            '--config_save_path', config_save_path,
            '--save_dir', save_dir,
            '--quiet',
            '--empty_cache'
        ] + (flags if flags is not None else [])

    @staticmethod
    def create_raw_interpret_args(dataset_type: str,
                                  checkpoint_dir: str,
                                  flags: List[str] = None) -> List[str]:
        """Creates a list of raw command line arguments for interpretation."""
        return [
            'interpret',  # Note: not actually used, just a placeholder
            '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_smiles.csv'),
            '--checkpoint_dir', checkpoint_dir
        ] + (flags if flags is not None else [])

    def train(self,
              dataset_type: str,
              metric: str,
              save_dir: str,
              model_type: str = 'chemprop',
              flags: List[str] = None):
        # Set up command line arguments
        raw_args = self.create_raw_train_args(
            dataset_type=dataset_type,
            metric=metric,
            save_dir=save_dir,
            model_type=model_type,
            flags=flags
        )

        # Train
        with patch('sys.argv', raw_args):
            command_line = ' '.join(raw_args[1:])

            if model_type == 'chemprop':
                print(f'python train.py {command_line}')
                chemprop_train()
            else:
                print(f'python sklearn_train.py {command_line}')
                sklearn_train()

    def predict(self,
                dataset_type: str,
                preds_path: str,
                save_dir: str,
                model_type: str = 'chemprop',
                flags: List[str] = None):
        # Set up command line arguments
        raw_args = self.create_raw_predict_args(
            dataset_type=dataset_type,
            preds_path=preds_path,
            checkpoint_dir=save_dir,
            flags=flags
        )

        # Predict
        with patch('sys.argv', raw_args):
            command_line = ' '.join(raw_args[1:])

            if model_type == 'chemprop':
                print(f'python predict.py {command_line}')
                chemprop_predict()
            else:
                print(f'python sklearn_predict.py {command_line}')
                sklearn_predict()

    def hyperopt(self,
                 dataset_type: str,
                 config_save_path: str,
                 save_dir: str,
                 flags: List[str] = None):
        # Set up command line arguments
        raw_args = self.create_raw_hyperopt_args(
            dataset_type=dataset_type,
            config_save_path=config_save_path,
            save_dir=save_dir,
            flags=flags
        )

        # Hyperopt
        with patch('sys.argv', raw_args):
            command_line = ' '.join(raw_args[1:])
            print(f'python hyperparameter_optimization.py {command_line}')
            chemprop_hyperopt()

    def interpret(self,
                  dataset_type: str,
                  checkpoint_dir: str,
                  flags: List[str] = None):
        # Set up command line arguments
        raw_args = self.create_raw_interpret_args(
            dataset_type=dataset_type,
            checkpoint_dir=checkpoint_dir,
            flags=flags
        )

        # Interpret
        with patch('sys.argv', raw_args):
            command_line = ' '.join(raw_args[1:])
            print(f'python interpret.py {command_line}')
            chemprop_interpret()

    def fingerprint(self,
                    dataset_type: str,
                    checkpoint_dir: str,
                    fingerprint_path: str,
                    fingerprint_flags: List[str]):
        # Set up command line arguments
        raw_args = self.create_raw_predict_args(
            dataset_type=dataset_type,
            preds_path=fingerprint_path,
            checkpoint_dir=checkpoint_dir,
            flags=fingerprint_flags
        )

        # Fingerprint
        with patch('sys.argv', raw_args):
            command_line = ' '.join(raw_args[1:])
            print(f'python fingerprint.py {command_line}')
            chemprop_fingerprint()

    @parameterized.expand([
        (
                'sklearn_random_forest',
                'random_forest',
                'rmse',
                1.582733
        ),
        (
                'sklearn_svm',
                'svm',
                'rmse',
                1.698927
        ),
        (
                'chemprop',
                'chemprop',
                'rmse',
                1.64048879,
        ),
        (
                'chemprop_scaffold_split',
                'chemprop',
                'rmse',
                1.70756238,
                ['--split_type', 'scaffold_balanced']
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                'rmse',
                1.99633537,
                ['--features_generator', 'morgan']
        ),
        (
                'chemprop_rdkit_features_path',
                'chemprop',
                'rmse',
                1.06655898,
                ['--features_path', os.path.join(TEST_DATA_DIR, 'regression.npz'), '--no_features_scaling']
        ),
        (
                'chemprop_bounded_mse_loss',
                'chemprop',
                'bounded_mse',
                2.9177008,
                [
                    '--loss_function', 'bounded_mse',
                    '--data_path', os.path.join(TEST_DATA_DIR, 'regression_inequality.csv')
                ]
        )
    ])
    def test_train_single_task_regression(self,
                                          name: str,
                                          model_type: str,
                                          metric: str,
                                          expected_score: float,
                                          train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            self.train(
                dataset_type='regression',
                metric=metric,
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])
            self.assertEqual(len(test_scores), 1)

            mean_score = np.mean(test_scores)
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA*expected_score)

    @parameterized.expand([
        (
                'chemprop',
                'chemprop',
                'auc',
                0.4908104,
                ['--class_balance', '--split_sizes', '0.4', '0.3', '0.3']
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                'auc',
                0.4733686,
                ['--features_generator', 'morgan', '--class_balance', '--split_sizes', '0.4', '0.3', '0.3']
        ),
        (
                'chemprop_rdkit_features_path',
                'chemprop',
                'auc',
                0.4573833,
                ['--features_path', os.path.join(TEST_DATA_DIR, 'classification.npz'), '--no_features_scaling', '--class_balance', '--split_sizes', '0.4', '0.3', '0.3']
        ),
        (
                'chemprop_mcc_metric',
                'chemprop',
                'mcc',
                0.0352518,
                ['--metric', 'mcc', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_common.csv'), '--class_balance']
        ),
        (
                'chemprop_f1_metric',
                'chemprop',
                'f1',
                0.02777778,
                ['--metric', 'f1', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_common.csv'), '--class_balance']
        ),
        (
                'chemprop_mcc_loss',
                'chemprop',
                'auc',
                0.74079357,
                ['--loss_function', 'mcc', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_common.csv'), '--class_balance']
        )
    ])
    def test_train_multi_task_classification(self,
                                             name: str,
                                             model_type: str,
                                             metric: str,
                                             expected_score: float,
                                             train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            self.train(
                dataset_type='classification',
                metric=metric,
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])
            mean_score = np.mean(np.array(test_scores))
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA*expected_score)

    @parameterized.expand([
        (
                'sklearn_random_forest',
                'random_forest',
                0.9455894
        ),
        (
                'sklearn_svm',
                'svm',
                1.0151356
        ),
        (
                'chemprop',
                'chemprop',
                1.1261400
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                4.1756080,
                ['--features_generator', 'morgan'],
                ['--features_generator', 'morgan']
        ),
        (
                'chemprop_rdkit_features_path',
                'chemprop',
                0.47390878,
                ['--features_path', os.path.join(TEST_DATA_DIR, 'regression.npz'), '--no_features_scaling'],
                ['--features_path', os.path.join(TEST_DATA_DIR, 'regression_test.npz'), '--no_features_scaling']
        )
    ])
    def test_predict_single_task_regression(self,
                                            name: str,
                                            model_type: str,
                                            expected_score: float,
                                            train_flags: List[str] = None,
                                            predict_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'regression'
            self.train(
                dataset_type=dataset_type,
                metric='rmse',
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(
                dataset_type=dataset_type,
                preds_path=preds_path,
                save_dir=save_dir,
                model_type=model_type,
                flags=predict_flags
            )

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.mean((pred - true) ** 2))
            self.assertAlmostEqual(mse, expected_score, delta=DELTA*expected_score)
    
    def test_predict_individual_ensemble(self):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'regression'
            self.train(
                dataset_type=dataset_type,
                metric='rmse',
                save_dir=save_dir,
            )

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(
                dataset_type=dataset_type,
                preds_path=preds_path,
                save_dir=save_dir,
                flags=['--individual_ensemble_predictions']
            )

            pred = pd.read_csv(preds_path)
            columns = list(pred.columns)
            expected_columns = ['smiles', 'logSolubility'] + [f'logSolubility_model_{idx}' for idx in range(NUM_FOLDS)]
            self.assertTrue(columns == expected_columns)


    @parameterized.expand([
        (
                'chemprop',
                'chemprop',
                0.1804146,
                ['--class_balance', '--split_sizes', '0.4', '0.3', '0.3'],
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                0.26773606,
                ['--features_generator', 'morgan', '--class_balance', '--split_sizes', '0.4', '0.3', '0.3'],
                ['--features_generator', 'morgan']
        ),
        (
                'chemprop_rdkit_features_path',
                'chemprop',
                0.0778546,
                ['--features_path', os.path.join(TEST_DATA_DIR, 'classification.npz'), '--no_features_scaling', '--class_balance', '--split_sizes', '0.4', '0.3', '0.3'],
                ['--features_path', os.path.join(TEST_DATA_DIR, 'classification_test.npz'), '--no_features_scaling']
        )
    ])
    def test_predict_multi_task_classification(self,
                                               name: str,
                                               model_type: str,
                                               expected_score: float,
                                               train_flags: List[str] = None,
                                               predict_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'classification'
            self.train(
                dataset_type=dataset_type,
                metric='auc',
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(
                dataset_type=dataset_type,
                preds_path=preds_path,
                save_dir=save_dir,
                model_type=model_type,
                flags=predict_flags
            )

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_true.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            mse = float(np.nanmean((pred - true) ** 2))
            self.assertAlmostEqual(mse, expected_score, delta=DELTA*expected_score)

    def test_chemprop_hyperopt(self):
        with TemporaryDirectory() as save_dir:
            # Train
            config_save_path = os.path.join(save_dir, 'config.json')
            self.hyperopt(
                dataset_type='regression',
                config_save_path=config_save_path,
                save_dir=save_dir
            )

            # Check results
            with open(config_save_path) as f:
                config = json.load(f)

            parameters = {'depth': (2, 6), 'hidden_size': (300, 2400), 'ffn_num_layers': (1, 3), 'dropout': (0.0, 0.4)}

            self.assertEqual(set(config.keys()), set(parameters.keys()))

            for parameter, (min_value, max_value) in parameters.items():
                self.assertTrue(min_value <= config[parameter] <= max_value)

    @parameterized.expand([
        (
                'chemprop',
        ),
        (
                'chemprop_morgan_features_generator',
                ['--features_generator', 'morgan'],
                ['--features_generator', 'morgan']
        ),
    ])
    def test_interpret_single_task_regression(self,
                                              name: str,
                                              train_flags: List[str] = None,
                                              interpret_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'regression'
            self.train(
                dataset_type=dataset_type,
                metric='rmse',
                save_dir=save_dir,
                flags=train_flags
            )

            # Interpret
            try:
                self.interpret(
                    dataset_type=dataset_type,
                    checkpoint_dir=save_dir,
                    flags=interpret_flags
                )
            except Exception as e:
                self.fail(f'Interpretation failed with error: {e}')

    def test_chemprop_web(self):
        with TemporaryDirectory() as root_dir:
            app = build_app(root_folder=root_dir, init_db=True)

            app.config['TESTING'] = True
            app.config['SERVER_NAME'] = 'localhost'

            data_path = 'regression.csv'
            test_path = 'regression_test_smiles.csv'
            dataset_name = 'regression_data'
            dataset_type = 'regression'
            checkpoint_name = 'regression_ckpt'
            ckpt_name = data_name = '1'
            epochs = 3
            ensemble_size = 1

            with open(os.path.join(TEST_DATA_DIR, data_path)) as f:
                train_data = BytesIO(f.read().encode('utf-8'))

            with open(os.path.join(TEST_DATA_DIR, test_path)) as f:
                test_smiles = f.read()

            with app.app_context():
                with app.test_client() as client:
                    response = client.get('/')
                    self.assertEqual(response.status_code, 200)

                    # Upload data
                    response = client.post(
                        url_for('upload_data', return_page='home'),
                        data={
                            'dataset': (train_data, data_path),
                            'datasetName': dataset_name
                        }
                    )
                    self.assertEqual(response.status_code, 302)

                    # Train
                    response = client.post(
                        url_for('train'),
                        data={
                            'dataName': data_name,
                            'epochs': epochs,
                            'ensembleSize': ensemble_size,
                            'checkpointName': checkpoint_name,
                            'datasetType': dataset_type,
                            'useProgressBar': False
                        }
                    )
                    self.assertEqual(response.status_code, 200)

                    # Predict
                    response = client.post(
                        url_for('predict'),
                        data={
                            'checkpointName': ckpt_name,
                            'textSmiles': test_smiles
                        }
                    )
                    self.assertEqual(response.status_code, 200)

    @parameterized.expand([
        (
            'spectra',
            'chemprop',
            0.00520246,
            [
                '--data_path', os.path.join(TEST_DATA_DIR, 'spectra.csv'),
                '--features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
                '--split_type', 'random_with_repeated_smiles'
            ]
        ),
        (
            'spectra_excluded_targets',
            'chemprop',
            0.003938459,
            [
                '--data_path', os.path.join(TEST_DATA_DIR, 'spectra_exclusions.csv'),
                '--features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
                '--split_type', 'random_with_repeated_smiles'
            ]
        ),
        (
            'spectra_phase_features',
            'chemprop',
            0.0065630322,
            [
                '--data_path', os.path.join(TEST_DATA_DIR, 'spectra_exclusions.csv'),
                '--phase_features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
                '--spectra_phase_mask_path', os.path.join(TEST_DATA_DIR, 'spectra_mask.csv'),
                '--split_type', 'random_with_repeated_smiles'
            ]
        ),
    ])
    def test_train_spectra(self,
                                          name: str,
                                          model_type: str,
                                          expected_score: float,
                                          train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'sid'
            self.train(
                dataset_type = 'spectra',
                metric = metric,
                save_dir = save_dir,
                model_type = model_type,
                flags = train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])
            self.assertEqual(len(test_scores), 1)

            mean_score = np.mean(test_scores)
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA*expected_score)

    @parameterized.expand([
        (
            'spectra',
            'chemprop',
            0.0041501114,
            0,
            [
                '--data_path', os.path.join(TEST_DATA_DIR, 'spectra.csv'),
                '--features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
                '--split_type', 'random_with_repeated_smiles'
            ],
            [
                '--features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
            ]
        ),
        (
            'spectra_phase_features',
            'chemprop',
            0.0053274466,
            284,
            [
                '--data_path', os.path.join(TEST_DATA_DIR, 'spectra_exclusions.csv'),
                '--phase_features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
                '--spectra_phase_mask_path', os.path.join(TEST_DATA_DIR, 'spectra_mask.csv'),
                '--split_type', 'random_with_repeated_smiles'
            ],
            [
                '--phase_features_path', os.path.join(TEST_DATA_DIR, 'spectra_features.csv'),
            ]
        ),
    ])
    def test_predict_spectra(self,
                                            name: str,
                                            model_type: str,
                                            expected_score: float,
                                            expected_nans: int,
                                            train_flags: List[str] = None,
                                            predict_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'spectra'
            self.train(
                dataset_type=dataset_type,
                metric='sid',
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Predict
            preds_path = os.path.join(save_dir, 'preds.csv')
            self.predict(
                dataset_type=dataset_type,
                preds_path=preds_path,
                save_dir=save_dir,
                model_type=model_type,
                flags=predict_flags
            )

            # Check results
            pred = pd.read_csv(preds_path)
            true = pd.read_csv(os.path.join(TEST_DATA_DIR, 'spectra.csv'))
            self.assertEqual(list(pred.keys()), list(true.keys()))
            self.assertEqual(list(pred['smiles']), list(true['smiles']))

            pred, true = pred.drop(columns=['smiles']), true.drop(columns=['smiles'])
            pred, true = pred.to_numpy(), true.to_numpy()
            phase_features = load_features(predict_flags[1])
            if '--spectra_phase_mask_path' in train_flags:
                mask = load_phase_mask(train_flags[5])
            else:
                mask = None
            true = normalize_spectra(true,phase_features,mask)
            sid = evaluate_predictions(preds=pred, targets=true, num_tasks=len(true[0]), metrics=['sid'], dataset_type='spectra')['sid'][0]
            self.assertAlmostEqual(sid, expected_score, delta=DELTA*expected_score)
            self.assertEqual(np.sum(np.isnan(pred)),expected_nans)

    @parameterized.expand([
        (
                'chemprop_reaction',
                'chemprop',
                2.1235725,
                ['--reaction', '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_regression.csv')]
        ),
        (
                'chemprop_scaffold_split',
                'chemprop',
                2.0610431,
                ['--reaction', '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_regression.csv'),'--split_type', 'scaffold_balanced']
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                2.8446566,
                ['--reaction', '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_regression.csv'),'--features_generator', 'morgan']
        ),
        (
                'chemprop_reaction_explicit_h',
                'chemprop',
                2.2980834,
                ['--reaction', '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_regression.csv'), '--explicit_h']
         )
    ])
    def test_train_single_task_regression_reaction(self,
                                          name: str,
                                          model_type: str,
                                          expected_score: float,
                                          train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'rmse'
            self.train(
                dataset_type = 'regression',
                metric = metric,
                save_dir = save_dir,
                model_type = model_type,
                flags = train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])
            self.assertEqual(len(test_scores), 1)

            mean_score = np.mean(test_scores)
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA*expected_score)

    @parameterized.expand([
        (
                'chemprop',
                'chemprop',
                'auc',
                0.699453,
                ['--number_of_molecules', '2', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_multimolecule.csv')]
        )
    ])
    def test_single_task_multimolecule_classification(self,
                                          name: str,
                                          model_type: str,
                                          metric: str,
                                          expected_score: float,
                                          train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            self.train(
                dataset_type='classification',
                metric=metric,
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])

            mean_score = np.mean(test_scores)
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA * expected_score)

    @parameterized.expand([
        (
                'chemprop',
                'chemprop',
                2805.4368779,
                ['--fingerprint_type', 'MPN'],
        ),
        (
                'chemprop',
                'chemprop',
                2689.55376,
                ['--fingerprint_type', 'last_FFN'],
        )
    ])
    def test_single_task_fingerprint(self,
                                            name: str,
                                            model_type: str,
                                            expected_score: float,
                                            fingerprint_flags: List[str],
                                            train_flags: List[str] = None,
                                     ):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'classification'
            self.train(
                dataset_type=dataset_type,
                metric='auc',
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Fingerprint
            fingerprint_path = os.path.join(save_dir, 'fingerprints.csv')
            self.fingerprint(
                dataset_type=dataset_type,
                checkpoint_dir=save_dir,
                fingerprint_path=fingerprint_path,
                fingerprint_flags=fingerprint_flags
            )

            fingerprints = pd.read_csv(fingerprint_path).drop(["smiles"], axis=1)
            self.assertAlmostEqual(np.sum(fingerprints.to_numpy()), expected_score, delta=DELTA*expected_score)

    @parameterized.expand([
        (
                'chemprop',
                'chemprop',
                ['--fingerprint_type', 'MPN'],
                True,
                ['--number_of_molecules', '2', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_multimolecule.csv')]
        ),
        (
                'chemprop',
                'chemprop',
                ['--fingerprint_type', 'MPN'],
                False,
                ['--number_of_molecules', '2', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_multimolecule.csv'), '--mpn_shared']
        ),
        (
                'chemprop',
                'chemprop',
                ['--fingerprint_type', 'last_FFN'],
                True,
                ['--number_of_molecules', '2', '--data_path', os.path.join(TEST_DATA_DIR, 'classification_multimolecule.csv')]
        )
    ])
    def test_multimolecule_fingerprint_with_single_input(self,
                                     name: str,
                                     model_type: str,
                                     fingerprint_flags: List[str],
                                     exception_thrown: bool,
                                     train_flags: List[str] = None
                                     ):
        with TemporaryDirectory() as save_dir:
            # Train
            dataset_type = 'classification'
            self.train(
                dataset_type=dataset_type,
                metric='auc',
                save_dir=save_dir,
                model_type=model_type,
                flags=train_flags
            )

            # Predict
            fingerprint_path = os.path.join(save_dir, 'fingerprints.csv')

            # Check to make sure that an exception is thrown for cases where the model isn't built with --mpn-shared and with a fingerprint
            # type of MPN
            if exception_thrown:
                with self.assertRaises(ValueError):
                    self.fingerprint(
                        dataset_type=dataset_type,
                        checkpoint_dir=save_dir,
                        fingerprint_path=fingerprint_path,
                        fingerprint_flags=fingerprint_flags
                    )
            else:
                self.fingerprint(
                    dataset_type=dataset_type,
                    checkpoint_dir=save_dir,
                    fingerprint_path=fingerprint_path,
                    fingerprint_flags=fingerprint_flags
                )
                fingerprints = pd.read_csv(fingerprint_path)
                test_input = pd.read_csv(os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_smiles.csv'))

                self.assertEqual(list(fingerprints['smiles'].values), list(test_input['smiles'].values))

    @parameterized.expand([(
                'chemprop_reaction_solvent',
                'chemprop',
                2.912189,
                ['--reaction_solvent', '--number_of_molecules', '2',
                 '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_solvent_regression.csv')]
        ),
        (
                'chemprop_morgan_features_generator',
                'chemprop',
                3.7687076,
                ['--reaction_solvent', '--number_of_molecules', '2',
                 '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_solvent_regression.csv'),'--features_generator', 'morgan']
        ),
        (
                'chemprop_reaction_solvent_explicit_h',
                'chemprop',
                2.805125,
                ['--reaction_solvent', '--number_of_molecules', '2',
                 '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_solvent_regression.csv'), '--explicit_h']
         ),
        (
                'chemprop_reaction_solvent_explicit_h_adding_h',
                'chemprop',
                2.8814398,
                ['--reaction_solvent', '--number_of_molecules', '2',
                 '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_solvent_regression.csv'), '--explicit_h', '--adding_h']
        ),
        (
                'chemprop_reaction_solvent_diff_mpn_size',
                'chemprop',
                2.9015592,
                ['--reaction_solvent', '--number_of_molecules', '2',
                 '--data_path', os.path.join(TEST_DATA_DIR, 'reaction_solvent_regression.csv'), '--hidden_size', '500',
                 '--hidden_size_solvent', '250']
        )
    ])
    def test_train_single_task_regression_reaction_solvent(self,
                                          name: str,
                                          model_type: str,
                                          expected_score: float,
                                          train_flags: List[str] = None):
        with TemporaryDirectory() as save_dir:
            # Train
            metric = 'rmse'
            self.train(
                dataset_type = 'regression',
                metric = metric,
                save_dir = save_dir,
                model_type = model_type,
                flags = train_flags
            )

            # Check results
            test_scores_data = pd.read_csv(os.path.join(save_dir, TEST_SCORES_FILE_NAME))
            test_scores = np.array(test_scores_data[f'Mean {metric}'])
            self.assertEqual(len(test_scores), 1)

            mean_score = np.mean(test_scores)
            self.assertAlmostEqual(mean_score, expected_score, delta=DELTA*expected_score)

    @parameterized.expand([(
        101.8037,
        'ensemble',
        None,
        'nll',
        [],
        [],
    ),
    (
        1.749264,
        'mve',
        None,
        'nll',
        ['--loss_function', 'mve'],
        [],
    ),
    (
        1.93262,
        'evidential_epistemic',
        None,
        'nll',
        ['--loss_function', 'evidential'],
        [],
    ),
    (
        1.936139,
        'evidential_aleatoric',
        None,
        'nll',
        ['--loss_function', 'evidential'],
        [],
    ),
    (
        1.9502012,
        'evidential_total',
        None,
        'nll',
        ['--loss_function', 'evidential'],
        [],
    ),
    # (
    #     8.843267,
    #     'dropout',
    #     None,
    #     'nll',
    #     ['--num_folds', '1'],
    #     [],
    # ),
    (
        2.350392723,
        'ensemble',
        'zscaling',
        'nll',
        [],
        [],
    ),
    (
        1.8707739,
        'ensemble',
        'tscaling',
        'nll',
        ['--num_folds','5'],
        [],
    ),
    (
        1.2245133,
        'ensemble',
        'zelikman_interval',
        'ence',
        [],
        [],
    ),
    (
        2.06247499,
        'mve',
        'mve_weighting',
        'nll',
        ['--loss_function', 'mve'],
        [],
    ),
    (
        0.38442,
        'ensemble',
        None,
        'miscalibration_area',
        [],
        [],
    ),
    (
        166.80048,
        'ensemble',
        None,
        'ence',
        [],
        [],
    ),
    # (
    #     0.0239197,
    #     'ensemble',
    #     None,
    #     'spearman',
    #     [],
    #     [],
    # ),
    ])
    def test_uncertainty_regression(
        self,
        expected_score: float,
        uncertainty_method: str,
        calibration_method: str,
        evaluation_methods: str,
        train_flags: List[str] = None,
        predict_flags: List[str] = None,
    ):
        with TemporaryDirectory() as save_dir:
            self.train(
                dataset_type='regression',
                metric='rmse',
                save_dir=save_dir,
                flags=train_flags,
            )
            test_path = os.path.join(TEST_DATA_DIR, 'regression.csv')
            eval_path = os.path.join(save_dir, 'eval_scores.csv')
            predict_flags.extend(['--evaluation_scores_path', eval_path, '--test_path', test_path])
            if uncertainty_method is not None:
                predict_flags.extend(['--uncertainty_method', uncertainty_method,])
            if calibration_method is not None:
                predict_flags.extend(['--calibration_method', calibration_method, '--calibration_path', test_path])
            if evaluation_methods is not None:
                predict_flags.extend(['--evaluation_methods', evaluation_methods])
            self.predict(
                dataset_type='regression',
                preds_path=os.path.join(save_dir, 'preds.csv'),
                save_dir=save_dir,
                flags=predict_flags,
            )
            evaluation_scores_data=pd.read_csv(eval_path)
            self.assertAlmostEqual(evaluation_scores_data['logSolubility'][0], expected_score, delta=expected_score * DELTA)

    @parameterized.expand([(
        0.62062329,
        'classification',
        'platt',
        'nll',
        ['--number_of_molecules', '2'],
        ['--number_of_molecules', '2'],
    ),
    (
        0.60075398,
        'classification',
        'isotonic',
        'nll',
        ['--number_of_molecules', '2'],
        ['--number_of_molecules', '2'],
    ),
    (
        0.67181467,
        'classification',
        'isotonic',
        'accuracy',
        ['--number_of_molecules', '2'],
        ['--number_of_molecules', '2'],
    ),
    ])
    def test_uncertainty_class(
        self,
        expected_score: float,
        uncertainty_method: str,
        calibration_method: str,
        evaluation_methods: str,
        train_flags: List[str] = None,
        predict_flags: List[str] = None,
    ):
        with TemporaryDirectory() as save_dir:
            test_path = os.path.join(TEST_DATA_DIR, 'classification_multimolecule.csv')
            train_flags.extend(['--data_path', test_path])
            self.train(
                dataset_type='classification',
                metric='binary_cross_entropy',
                save_dir=save_dir,
                flags=train_flags,
            )
            eval_path = os.path.join(save_dir, 'eval_scores.csv')
            predict_flags.extend(['--evaluation_scores_path', eval_path, '--test_path', test_path])
            if uncertainty_method is not None:
                predict_flags.extend(['--uncertainty_method', uncertainty_method,])
            if calibration_method is not None:
                predict_flags.extend(['--calibration_method', calibration_method, '--calibration_path', test_path])
            if evaluation_methods is not None:
                predict_flags.extend(['--evaluation_methods', evaluation_methods])
            self.predict(
                dataset_type='regression',
                preds_path=os.path.join(save_dir, 'preds.csv'),
                save_dir=save_dir,
                flags=predict_flags,
            )
            evaluation_scores_data=pd.read_csv(eval_path)
            self.assertAlmostEqual(evaluation_scores_data['synergy'][0], expected_score, delta=expected_score * DELTA)


if __name__ == '__main__':
    unittest.main()
