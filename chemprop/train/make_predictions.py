from argparse import Namespace
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem

from .predict import predict
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers


def make_predictions(args: Namespace, smiles: List[str] = None, invalid_smiles_warning: str = None) -> List[List[float]]:
    """Makes predictions."""
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    if invalid_smiles_warning is not None:
        success_indices = []
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                success_indices.append(i)
        full_smiles = smiles
        smiles = [smiles[i] for i in success_indices]

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles)
    else:
        test_data = get_data(args.test_path, args, use_compound_names=args.compound_names)
    test_smiles = test_data.smiles()
    if args.compound_names:
        compound_names = test_data.compound_names()
    print('Test size = {:,}'.format(len(test_data)))

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    sum_preds = np.zeros((len(test_data), args.num_tasks))
    print('Predicting with an ensemble of {} models'.format(len(args.checkpoint_paths)))
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
    print('Saving predictions to {}'.format(args.preds_path))

    with open(args.preds_path, 'w') as f:
        if args.write_smiles:
            f.write('smiles,')
        if args.compound_names:
            f.write('compound_name,')
        f.write(','.join(args.task_names) + '\n')

        for i in range(len(avg_preds)):
            if args.write_smiles:
                f.write(test_smiles[i] + ',')
            if args.compound_names:
                f.write(compound_names[i] + ',')
            f.write(','.join(str(p) for p in avg_preds[i]) + '\n')

    if invalid_smiles_warning is not None:
        full_preds = [[invalid_smiles_warning] for _ in range(len(full_smiles))]
        for i, si in enumerate(success_indices):
            full_preds[si] = avg_preds[i]
        return full_preds

    return avg_preds
