import os
import re
import csv
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from .predict import predict
from .bayesgrad import bayes_predict
from chemprop.args import PredictArgs, TrainArgs

from chemprop.data.utils import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit
from chemprop.visuals.bayesgrad_visualizer import BayesEnsembleVisualizer

@timeit()
def make_predictions(args: PredictArgs,
                     smiles: List[List[str]] = None,
                     bayes_ensemble_grad: bool = False) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.


    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param bayes_ensemble_grad: Whether to perform local interpretation using BayesEnsembleGrad
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
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
        full_data = get_data(path=args.test_path, target_columns=[], ignore_columns=[], skip_invalid_smiles=False,
                             args=args, store_row=True)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Average gradients returned are defaulted to None for each SMILES for each target
    avg_grads = {target_id: [None] * len(full_data) for target_id in range(num_tasks)}

    # Explanation plots. First rows are SMILES, columns are targets
    explanation_plots = np.full((len(full_data), num_tasks), fill_value=None)

    # Initially we don't know the shape of the gradients;
    # for each molecule, we expect the sum of gradients for each bond and atom
    sum_grads = None

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

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
        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler = load_scalers(checkpoint_path)

        # Normalize features
        if args.features_scaling:
            test_data.reset_features_and_targets()
            test_data.normalize_features(features_scaler)

        if not bayes_ensemble_grad:
            model_preds = predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler,
            )
        else:
            model_preds, model_grads = bayes_predict(
                model=model,
                data_loader=test_data_loader,
                scaler=scaler,
                args=args
            )

            if sum_grads is None:
                sum_grads = model_grads
            else:
                # First level in the returned dictionary maps each target to the gradients data
                for target_id, batch_grad_agg in sum_grads.items():
                    # Second level in the returned dictionary maps each molecule to the gradients data
                    for molecule_id, molecule_grad_agg in enumerate(batch_grad_agg):
                        # Add the aggregated gradients for each atom
                        sum_grads[target_id][molecule_id]['atoms'] = {k: v + molecule_grad_agg['atoms'][k] for k, v in sum_grads[target_id][molecule_id]['atoms'].items()}
                        # Add the aggregated gradients for each bond
                        sum_grads[target_id][molecule_id]['bonds'] = {k: v + molecule_grad_agg['bonds'][k] for k, v in sum_grads[target_id][molecule_id]['bonds'].items()}

        # Sum up the predictions for each model
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    total_evaluations = len(args.checkpoint_paths)
    avg_preds = sum_preds / total_evaluations
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

    # Ensemble gradients - i.e. divide with the total number of models evaluated
    if bayes_ensemble_grad:

        # Get the valid indices
        valid_indices = [i for i, datapoint in enumerate(full_data) if full_to_valid_indices.get(i, None) is not None]

        # Get the target names
        target_names = [re.sub('[\W_]+', '', name, flags=re.UNICODE) for name in list(full_data[0].row.keys())[1:]]

        # First level in the returned dictionary maps each target to the gradients data
        for target_id, batch_grad_agg in avg_grads.items():
            # Second level in the returned dictionary maps each molecule to the gradients data
            # Invalid SMILES entries are already None, so we only update the valid ones
            for i, si in enumerate(valid_indices):
                # if full_to_valid_indices.get(full_index, None) is
                # This molecule is valid, so get average gradients
                avg_grads[target_id][si] = {
                    'atoms': {k: v / total_evaluations for k, v in sum_grads[target_id][i]['atoms'].items()},
                    'bonds': {k: v / total_evaluations for k, v in sum_grads[target_id][i]['bonds'].items()}
                }

        # Object we'll use to generate the visuals
        makedirs(args.bayes_path)
        drawer = BayesEnsembleVisualizer()

        # For each target we'll create separate images
        for target_id, target_name in zip(range(num_tasks), target_names):

            # Create folder for target
            save_path = os.path.join(args.bayes_path, target_name)
            makedirs(save_path)

            # For each valid SMILES string we'll create a visual
            smiles = full_data.smiles()
            for i, si in enumerate(valid_indices):
                svg = drawer.visualize(smiles[si], avg_grads[target_id][si])
                if svg:
                    with open(os.path.join(save_path, f'{si}.svg'), 'w') as f:
                        f.write(svg)

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
