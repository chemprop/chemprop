"""Helper/utility functions for Chemprop integration tests."""
import os
from typing import List

from unittest.mock import patch

from chemprop.hyperparameter_optimization import chemprop_hyperopt
from chemprop.interpret import chemprop_interpret
from chemprop.sklearn_predict import sklearn_predict
from chemprop.sklearn_train import sklearn_train
from chemprop.train import chemprop_train, chemprop_predict, chemprop_fingerprint

TEST_DATA_DIR = 'tests/data'
SEED = 0
EPOCHS = 3
NUM_FOLDS = 3
NUM_ITER = 2
DELTA = 0.015


def _create_raw_train_args(dataset_type: str,
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

def _create_raw_predict_args(dataset_type: str,
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

def _create_raw_hyperopt_args(dataset_type: str,
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

def _create_raw_interpret_args(dataset_type: str,
                               checkpoint_dir: str,
                               flags: List[str] = None) -> List[str]:
    """Creates a list of raw command line arguments for interpretation."""
    return [
        'interpret',  # Note: not actually used, just a placeholder
        '--data_path', os.path.join(TEST_DATA_DIR, f'{dataset_type}_test_smiles.csv'),
        '--checkpoint_dir', checkpoint_dir
    ] + (flags if flags is not None else [])

def train(dataset_type: str,
          metric: str,
          save_dir: str,
          model_type: str = 'chemprop',
          flags: List[str] = None):
    # Set up command line arguments
    raw_args = _create_raw_train_args(
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

def predict(dataset_type: str,
            preds_path: str,
            save_dir: str,
            model_type: str = 'chemprop',
            flags: List[str] = None):
    # Set up command line arguments
    raw_args = _create_raw_predict_args(
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

def hyperopt(dataset_type: str,
             config_save_path: str,
             save_dir: str,
             flags: List[str] = None):
    # Set up command line arguments
    raw_args = _create_raw_hyperopt_args(
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

def interpret(dataset_type: str,
              checkpoint_dir: str,
              flags: List[str] = None):
    # Set up command line arguments
    raw_args = _create_raw_interpret_args(
        dataset_type=dataset_type,
        checkpoint_dir=checkpoint_dir,
        flags=flags
    )

    # Interpret
    with patch('sys.argv', raw_args):
        command_line = ' '.join(raw_args[1:])
        print(f'python interpret.py {command_line}')
        chemprop_interpret()

def fingerprint(dataset_type: str,
                checkpoint_dir: str,
                fingerprint_path: str,
                fingerprint_flags: List[str]):
    # Set up command line arguments
    raw_args = _create_raw_predict_args(
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
