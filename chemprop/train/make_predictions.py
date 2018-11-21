from argparse import Namespace

import numpy as np
from tqdm import tqdm

from .predict import predict
from chemprop.data.utils import get_data
from chemprop.utils import load_args, load_checkpoint, load_scalers


def make_predictions(args: Namespace):
    """Makes predictions."""
    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    test_data = get_data(args.test_path, args, use_compound_names=args.compound_names)
    if args.compound_names:
        compound_names = test_data.compound_names()
    print('Test size = {:,}'.format(len(test_data)))

    # Normalize features
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
        if args.compound_names:
            f.write('compound_name,')
        f.write(','.join(args.task_names) + '\n')

        for i in range(len(avg_preds)):
            if args.compound_names:
                f.write(compound_names[i] + ',')
            f.write(','.join(str(p) for p in avg_preds[i]) + '\n')
