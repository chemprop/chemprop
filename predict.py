import os
from pprint import pprint

import numpy as np
import torch

from model import build_model
from nn_utils import param_count
from parsing import get_parser, modify_args
from train_utils import predict
from utils import get_data, get_task_names, StandardScaler


# TODO: SUPPORT ALL OPTIONS THAT TRAIN DOES
def make_predictions(args):
    """Makes predictions."""
    pprint(vars(args))

    print('Loading data')
    task_names = get_task_names(args.train_path)
    train_data = get_data(args.train_path)
    if args.compound_names:
        compound_names, test_data = get_data(args.test_path, use_compound_names=True)
    else:
        test_data = get_data(args.test_path)

    num_tasks = len(train_data[0][1])
    args.num_tasks = num_tasks

    print('Train size = {:,} | test size = {:,}'.format(
        len(train_data),
        len(test_data))
    )
    print('Number of tasks = {}'.format(num_tasks))

    # Initialize scaler and scaler training labels by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        print('Fitting scaler')
        _, train_labels = zip(*train_data)
        scaler = StandardScaler().fit(train_labels)
    else:
        scaler = None

    # Build model
    print('Building model')
    model = build_model(num_tasks, args)

    print(model)
    print('Number of parameters = {:,}'.format(param_count(model)))
    if args.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    # Predict on test set
    test_smiles, _ = zip(*test_data)
    sum_preds = np.zeros((len(test_smiles), num_tasks))

    # Predict with each model individually
    for checkpoint_path in args.checkpoint_paths:
        # Load state dict from checkpoint
        model.load_state_dict(torch.load(checkpoint_path))
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
    parser = get_parser()
    # TODO: Save the scaler so that training data doesn't need to be reloaded
    parser.add_argument('--train_path', type=str, required=True,
                        help='Path to CSV file containing training data')
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Save compound name in addition to predicted values (use when test file has compound names)')
    # Note: Need to specify `--checkpoint_dir`
    args = parser.parse_args()

    # Create directory for save_path
    save_dir = os.path.dirname(args.save_path)
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)

    modify_args(args)

    make_predictions(args)
