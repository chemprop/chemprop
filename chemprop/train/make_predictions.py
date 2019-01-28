from argparse import Namespace
import csv
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem

from .predict import predict
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers


def make_predictions(args: Namespace, smiles: List[str] = None, allow_invalid_smiles: bool = True) -> List[List[float]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :param allow_invalid_smiles: Whether to allow invalid smiles. If true, predictions for the invalid smiles
    are replaced with None.
    NOTE: Currently only works when smiles are provided as an argument instead of as a data file.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    if allow_invalid_smiles:
        assert smiles is not None  # Note: Currently only works with smiles provided, not with data file.
        print('Validating SMILES')
        valid_indices = []
        for i, s in tqdm(enumerate(smiles), total=len(smiles)):
            if Chem.MolFromSmiles(s) is not None and Chem.MolFromSmiles(s).GetNumHeavyAtoms() > 0:
                valid_indices.append(i)
        full_smiles = smiles
        smiles = [smiles[i] for i in valid_indices]

    # Edge case if empty list of smiles is provided
    if smiles is not None and len(smiles) == 0:
        return [None] * len(full_smiles) if allow_invalid_smiles else []

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.compound_names)
    test_smiles = test_data.smiles()

    if args.compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    sum_preds = np.zeros((len(test_data), args.num_tasks))
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            data=test_data,
            args=args,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    avg_preds = sum_preds / args.ensemble_size
    avg_preds = avg_preds.tolist()

    # Save predictions
    assert len(test_data) == len(avg_preds)
    print(f'Saving predictions to {args.preds_path}')

    # Put Nones for invalid smiles
    if allow_invalid_smiles:
        full_preds = [None] * len(full_smiles)
        for i, si in enumerate(valid_indices):
            full_preds[si] = avg_preds[i]
        avg_preds = full_preds
        test_smiles = full_smiles

    # Write predictions
    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)

        header = []
        header.append('smiles')
        if args.compound_names:
            header.append('compound_names')

        header.extend(args.task_names)
        writer.writerow(header)

        for i in range(len(avg_preds)):
            row = []

            row.append(test_smiles[i])
            if args.compound_names:
                row.append(compound_names[i])

            if avg_preds[i] is not None:
                row.extend(avg_preds[i])
            else:
                row.extend([''] * (args.num_tasks - 1))

            writer.writerow(row)

    return avg_preds
