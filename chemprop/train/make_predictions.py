import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .predict import predict
from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit


@timeit()
def make_predictions(args: PredictArgs, smiles: List[str] = None) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: SMILES to make predictions on.
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # If features were used during training, they must be used when predicting
    if ((train_args.features_path is not None or train_args.features_generator is not None)
            and args.features_path is None
            and args.features_generator is None):
        raise ValueError('Features were used during training so they must be specified again during prediction '
                         'using the same type of features as before (with either --features_generator or '
                         '--features_path and using --no_features_scaling if applicable).')

    # Update predict args with training arguments to create a merged args object
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[PredictArgs, TrainArgs]

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(
            path=args.test_path,
            args=args,
            target_columns=[],
            ignore_columns=[],
            skip_invalid_smiles=False,
            store_row=True
        )

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if full_data[full_index].mol is not None:
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, device=args.device)
        model_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    makedirs(args.preds_path, isfile=True)

    # Get prediction column names
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    else:
        task_names = task_names

    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)

        for pred_name, pred in zip(task_names, preds):
            datapoint.row[pred_name] = pred

    # Save
    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()

        for datapoint in full_data:
            writer.writerow(datapoint.row)

    return avg_preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
