from argparse import ArgumentParser, Namespace
import os

import numpy as np
import torch
from tqdm import tqdm

from parsing import add_predict_args, update_args_from_checkpoint_dir
from train_utils import predict
from utils import get_data, load_checkpoint


def make_predictions(args: Namespace):
    """Makes predictions."""
    print('Loading data')
    test_data = get_data(args.test_path, use_compound_names=args.compound_names)
    if args.compound_names:
        compound_names = test_data.compound_names()
    print('Test size = {:,}'.format(len(test_data)))

    # Predict on test set
    sum_preds = None

    # Predict with each model individually
    print('Predicting with an ensemble of {} models'.format(len(args.checkpoint_paths)))
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model, scaler, train_args = load_checkpoint(checkpoint_path, cuda=args.cuda, get_scaler=True, get_args=True)

        # Update args
        args.dataset_type, args.num_tasks, args.task_names, args.features = \
            train_args.dataset_type, train_args.num_tasks, train_args.task_names, train_args.features

        if sum_preds is None:
            sum_preds = np.zeros((len(test_data), args.num_tasks))

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
        if args.compound_names:
            f.write('compound_name,')
        f.write(','.join(args.task_names) + '\n')

        for i in range(len(avg_preds)):
            if args.compound_names:
                f.write(compound_names[i] + ',')
            f.write(','.join(str(p) for p in avg_preds[i]) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    add_predict_args(parser)
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
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

    # Create directory for preds path
    preds_dir = os.path.dirname(args.preds_path)
    if preds_dir != '':
        os.makedirs(preds_dir, exist_ok=True)

    make_predictions(args)
