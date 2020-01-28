from argparse import Namespace
import csv
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers

def validate_data(incoming_data: List[str]) -> Tuple[List[Optional[List[float]]], MoleculeDataset, List[Optional[List[float]]]]:
    """
    Validate a list of SMILES and create a MoleculeDataset of the valid SMILES

    :param incoming_data: SMILES input
    :return: full_data All incoming data
    :return: valid_data Valid SMILES as MoleculeDataset
    :return: valid_indices Indices of valid SMILES in full_data
    """
    print('Validating SMILES')
    valid_indices = [i for i in range(
        len(incoming_data)) if incoming_data[i].mol is not None]
    full_data = incoming_data
    valid_data = MoleculeDataset([incoming_data[i] for i in valid_indices])
    return full_data, valid_data, valid_indices


def fill_missing_data(args: Namespace, avg_preds: List[List[float]], full_data: MoleculeDataset, valid_indices: List[List[bool]]):
    full_preds = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
    avg_preds = full_preds
    test_smiles = full_data.smiles()
    if args.use_compound_names:
        compound_names = full_data.compound_names()
    else:
        compound_names = None
    return test_smiles, full_preds, compound_names


def write_predictions(args: Namespace, preds: List[List[float]], smiles: MoleculeDataset, compound_names: List[str]=None):
    """
    Write predictions to a file specificed in the arguments

    :param args: Arguments
    :param args: preds Predictions for each molecule (averaged over the ensemble)
    :param args: smiles SMILES for the predicted molecules
    """
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []

        if args.use_compound_names:
            header.append('compound_names')

        header.append('smiles')

        if args.dataset_type == 'multiclass':
            for name in args.task_names:
                for i in range(args.multiclass_num_classes):
                    header.append(name + '_class' + str(i))
        else:
            header.extend(args.task_names)
        writer.writerow(header)

        for i in range(len(preds)):
            row = []

            if args.use_compound_names:
                row.append(compound_names[i])

            row.append(smiles[i])

            if preds[i] is not None:
                if args.dataset_type == 'multiclass':
                    for task_probs in preds[i]:
                        row.extend(task_probs)
                else:
                    row.extend(preds[i])
            else:
                if args.dataset_type == 'multiclass':
                    row.extend([''] * args.num_tasks * args.multiclass_num_classes)
                else:
                    row.extend([''] * args.num_tasks)

            writer.writerow(row)

def make_predictions(args: Namespace, smiles: List[str] = None, return_variance: Optional[bool] = False) -> Tuple[List[Optional[List[float]]], Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: values A list of lists of target predictions.
    :return: variances (optional) A list of lists of target prediction variances.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, args=args)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    # Validate data
    full_data, test_data, valid_indices = validate_data(test_data)

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    
    for ii, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        x = np.asarray(model_preds)
        if ii == 0:
            m = np.asarray(x)
            v = np.zeros(x.shape)
        else:            
            _m = m.copy()
            m = _m + (x - _m) / (ii +1)
            v = v + (x - _m) * (x - m)
            assert m.shape == x.shape
            assert v.shape == x.shape
    
    # Ensemble predictions
    avg_preds = m
    avg_preds = avg_preds.tolist()

    # Save predictions
    assert len(test_data) == len(avg_preds)
    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    test_smiles, avg_preds, compound_names = fill_missing_data(args, avg_preds, full_data, valid_indices)

     # Write predictions
    if args.preds_path:
        write_predictions(args, test_smiles, avg_preds, compound_names)
    
    if return_variance:
        return avg_preds, v
    else:
        return avg_preds
