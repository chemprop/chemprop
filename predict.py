from argparse import ArgumentParser, Namespace
import os

import numpy as np
import torch
from tqdm import tqdm

from parsing import update_args_from_checkpoint_dir
from train_utils import predict
from utils import get_data, get_task_names, load_checkpoint


def make_predictions(args: Namespace):
    """Makes predictions."""
    print('Loading data')
    task_names = get_task_names(args.test_path)

    if args.compound_names:
        compound_names, test_data = get_data(args.test_path, use_compound_names=True)
    else:
        test_data = get_data(args.test_path)

    args.num_tasks = len(test_data[0][1])

    print('Test size = {:,}'.format(len(test_data)))
    print('Number of tasks = {}'.format(args.num_tasks))

    # Predict on test set
    test_smiles, _ = zip(*test_data)
    sum_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Predict with each model individually
    print('Predicting with an ensemble of {} models'.format(len(args.checkpoint_paths)))
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        model, scaler = load_checkpoint(checkpoint_path, get_scaler=True)

        if args.cuda:
            print('Moving model to cuda')
            model = model.cuda()

        model_preds = predict(
            model=model,
            smiles=test_smiles,
            args=args,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    avg_preds = sum_preds / args.ensemble_size
    avg_preds = avg_preds.tolist()

    # Save predictions
    assert len(test_smiles) == len(avg_preds)
    print('Saving predictions to {}'.format(args.save_path))

    with open(args.save_path, 'w') as f:
        f.write(','.join(task_names) + '\n')

        for i in range(len(avg_preds)):
            if args.compound_names:
                f.write(compound_names[i] + ',')
            f.write(','.join(str(p) for p in avg_preds[i]) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['classification', 'regression', 'regression_with_binning'],
                        help='Type of dataset, i.e. classification (cls) or regression (reg).'
                             'This determines the loss function used during training.')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    args = parser.parse_args()

    # Get checkpoint paths
    update_args_from_checkpoint_dir(args)

    # Cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    # Create directory for save_path
    save_dir = os.path.dirname(args.save_path)
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    make_predictions(args)
