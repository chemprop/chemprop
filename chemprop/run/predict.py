from argparse import Namespace
import numpy as np
from tqdm import tqdm
from chemprop.run.train_utils import predict
from chemprop.utils.utils import get_data, load_checkpoint
import os


def get_ensemble_predictions(args: Namespace):
    print('Loading training args')
    _, train_args = load_checkpoint(args.checkpoint_paths[0], get_args=True)

    # Update current args from training args without overwriting any values
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    test_data = get_data(args.test_path, args, use_compound_names=args.compound_names)
    print('Test size = {:,}'.format(len(test_data)))

    # Predict with each model individually
    ensemble_size = len(args.checkpoint_paths)
    if ensemble_size != args.ensemble_size:
        print("ensemble size does not match arguments {} {}".format(ensemble_size, args.ensemble_size))
    preds = np.zeros((len(test_data), args.num_tasks, ensemble_size))

    print('Predicting with an ensemble of {} models'.format(ensemble_size))
    for ii, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=ensemble_size)):
        # Load model
        model, scaler, train_args = load_checkpoint(checkpoint_path, cuda=args.cuda, get_scaler=True, get_args=True)
        model_preds = predict(
            model=model,
            data=test_data,
            args=args,
            scaler=scaler
        )
        preds[:, :, ii] = np.array(model_preds)
    return np.asarray(preds)


def make_predictions(args: Namespace):
    """Makes predictions."""
    # Create directory for preds path
    preds_dir = os.path.dirname(args.preds_path)
    if preds_dir != '':
        os.makedirs(preds_dir, exist_ok=True)
    # Ensemble predictions
    #    avg_preds = sum_preds / args.ensemble_size
    avg_preds = get_ensemble_predictions(args).mean(axis=2)
    avg_preds = avg_preds.tolist()

    print('Loading data')
    test_data = get_data(args.test_path, args, use_compound_names=args.compound_names)
    if args.compound_names:
        compound_names = test_data.compound_names()

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
    return avg_preds
