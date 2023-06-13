from collections import OrderedDict
import csv
from typing import List, Optional, Union, Tuple

import numpy as np

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset, StandardScaler, AtomBondScaler
from chemprop.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim, set_reaction, set_explicit_h, set_adding_hs, set_keeping_atom_map, reset_featurization_parameters
from chemprop.models import MoleculeModel
from chemprop.uncertainty import UncertaintyCalibrator, build_uncertainty_calibrator, UncertaintyEstimator, build_uncertainty_evaluator
from chemprop.multitask_utils import reshape_values


def load_model(args: PredictArgs, generator: bool = False):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler 
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or 
                 generator object of scalers, the number of tasks and their respective names.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    # Load model and scalers
    models = (
        load_checkpoint(checkpoint_path, device=args.device) for checkpoint_path in args.checkpoint_paths
    )
    scalers = (
        load_scalers(checkpoint_path) for checkpoint_path in args.checkpoint_paths
    )
    if not generator:
        models = list(models)
        scalers = list(scalers)

    return args, train_args, models, scalers, num_tasks, task_names


def load_data(args: PredictArgs, smiles: List[List[str]]):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints, a :class:`~chemprop.data.MoleculeDataset` containing only valid datapoints,
                 a :class:`~chemprop.data.MoleculeDataLoader` and a dictionary mapping full to valid indices.
    """
    print("Loading data")
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator,
        )
    else:
        full_data = get_data(
            path=args.test_path,
            smiles_columns=args.smiles_columns,
            target_columns=[],
            ignore_columns=[],
            skip_invalid_smiles=False,
            args=args,
            store_row=not args.drop_extra_columns,
        )

    print("Validating SMILES")
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset(
        [full_data[i] for i in sorted(full_to_valid_indices.keys())]
    )

    print(f"Test size = {len(test_data):,}")

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    return full_data, test_data, test_data_loader, full_to_valid_indices


def set_features(args: PredictArgs, train_args: TrainArgs):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    reset_featurization_parameters()

    if args.atom_descriptors == "feature":
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_descriptors == "feature":
        set_extra_bond_fdim(train_args.bond_features_size)

    # set explicit H option and reaction option
    set_explicit_h(train_args.explicit_h)
    set_adding_hs(args.adding_h)
    set_keeping_atom_map(args.keeping_atom_map)
    if train_args.reaction:
        set_reaction(train_args.reaction, train_args.reaction_mode)
    elif train_args.reaction_solvent:
        set_reaction(True, train_args.reaction_mode)


def predict_and_save(
    args: PredictArgs,
    train_args: TrainArgs,
    test_data: MoleculeDataset,
    task_names: List[str],
    num_tasks: int,
    test_data_loader: MoleculeDataLoader,
    full_data: MoleculeDataset,
    full_to_valid_indices: dict,
    models: List[MoleculeModel],
    scalers: List[Union[StandardScaler, AtomBondScaler]],
    num_models: int,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = False,
    save_results: bool = True,
):
    """
    Function to predict with a model and save the predictions to file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    :param test_data: A :class:`~chemprop.data.MoleculeDataset` containing valid datapoints.
    :param task_names: A list of task names.
    :param num_tasks: Number of tasks.
    :param test_data_loader: A :class:`~chemprop.data.MoleculeDataLoader` to load the test data.
    :param full_data:  A :class:`~chemprop.data.MoleculeDataset` containing all (valid and invalid) datapoints.
    :param full_to_valid_indices: A dictionary dictionary mapping full to valid indices.
    :param models: A list or generator object of :class:`~chemprop.models.MoleculeModel`\ s.
    :param scalers: A list or generator object of :class:`~chemprop.features.scaler.StandardScaler` objects.
    :param num_models: The number of models included in the models and scalers input.
    :param calibrator: A :class: `~chemprop.uncertainty.UncertaintyCalibrator` object, for use in calibrating uncertainty predictions.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :param save_results: Whether to save the predictions in a csv. Function returns the predictions regardless.
    :return: A list of lists of target predictions.
    """
    estimator = UncertaintyEstimator(
        test_data=test_data,
        test_data_loader=test_data_loader,
        uncertainty_method=args.uncertainty_method,
        models=models,
        scalers=scalers,
        num_models=num_models,
        dataset_type=args.dataset_type,
        loss_function=args.loss_function,
        uncertainty_dropout_p=args.uncertainty_dropout_p,
        dropout_sampling_size=args.dropout_sampling_size,
        individual_ensemble_predictions=args.individual_ensemble_predictions,
        spectra_phase_mask=getattr(train_args, "spectra_phase_mask", None),
    )

    preds, unc = estimator.calculate_uncertainty(
        calibrator=calibrator
    )  # preds and unc are lists of shape(data,tasks)

    if calibrator is not None and args.is_atom_bond_targets and args.calibration_method == "isotonic":
        unc = reshape_values(unc, test_data, len(args.atom_targets), len(args.bond_targets), num_tasks)

    if args.individual_ensemble_predictions:
        individual_preds = (
            estimator.individual_predictions()
        )  # shape(data, tasks, ensemble) or (data, tasks, classes, ensemble)

    if args.evaluation_methods is not None:

        evaluation_data = get_data(
            path=args.test_path,
            smiles_columns=args.smiles_columns,
            target_columns=task_names,
            args=args,
            features_path=args.features_path,
            features_generator=args.features_generator,
            phase_features_path=args.phase_features_path,
            atom_descriptors_path=args.atom_descriptors_path,
            bond_descriptors_path=args.bond_descriptors_path,
            max_data_size=args.max_data_size,
            loss_function=args.loss_function,
        )

        evaluators = []
        for evaluation_method in args.evaluation_methods:
            evaluator = build_uncertainty_evaluator(
                evaluation_method=evaluation_method,
                calibration_method=args.calibration_method,
                uncertainty_method=args.uncertainty_method,
                dataset_type=args.dataset_type,
                loss_function=args.loss_function,
                calibrator=calibrator,
                is_atom_bond_targets=args.is_atom_bond_targets,
            )
            evaluators.append(evaluator)
    else:
        evaluators = None

    if evaluators is not None:
        evaluations = []
        print(f"Evaluating uncertainty for tasks {task_names}")
        for evaluator in evaluators:
            evaluation = evaluator.evaluate(
                targets=evaluation_data.targets(), preds=preds, uncertainties=unc, mask=evaluation_data.mask()
            )
            evaluations.append(evaluation)
            print(
                f"Using evaluation method {evaluator.evaluation_method}: {evaluation}"
            )
    else:
        evaluations = None

    # Save results
    if save_results:
        print(f"Saving predictions to {args.preds_path}")
        assert len(test_data) == len(preds)
        assert len(test_data) == len(unc)

        makedirs(args.preds_path, isfile=True)

        # Set multiclass column names, update num_tasks definitions
        if args.dataset_type == "multiclass":
            original_task_names = task_names
            task_names = [
                f"{name}_class_{i}"
                for name in task_names
                for i in range(args.multiclass_num_classes)
            ]
            num_tasks = num_tasks * args.multiclass_num_classes
        if args.uncertainty_method == "spectra_roundrobin":
            num_unc_tasks = 1
        else:
            num_unc_tasks = num_tasks

        # Copy predictions over to full_data
        for full_index, datapoint in enumerate(full_data):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                d_preds = preds[valid_index]
                d_unc = unc[valid_index]
                if args.individual_ensemble_predictions:
                    ind_preds = individual_preds[valid_index]
            else:
                d_preds = ["Invalid SMILES"] * num_tasks
                d_unc = ["Invalid SMILES"] * num_unc_tasks
                if args.individual_ensemble_predictions:
                    ind_preds = [["Invalid SMILES"] * len(args.checkpoint_paths)] * num_tasks
            # Reshape multiclass to merge task and class dimension, with updated num_tasks
            if args.dataset_type == "multiclass":
                d_preds = np.array(d_preds).reshape((num_tasks))
                d_unc = np.array(d_unc).reshape((num_unc_tasks))
                if args.individual_ensemble_predictions:
                    ind_preds = ind_preds.reshape(
                        (num_tasks, len(args.checkpoint_paths))
                    )

            # If extra columns have been dropped, add back in SMILES columns
            if args.drop_extra_columns:
                datapoint.row = OrderedDict()

                smiles_columns = args.smiles_columns

                for column, smiles in zip(smiles_columns, datapoint.smiles):
                    datapoint.row[column] = smiles

            # Add predictions columns
            if args.uncertainty_method == "spectra_roundrobin":
                unc_names = [estimator.label]
            else:
                unc_names = [name + f"_{estimator.label}" for name in task_names]

            for pred_name, unc_name, pred, un in zip(
                task_names, unc_names, d_preds, d_unc
            ):
                datapoint.row[pred_name] = pred
                if args.uncertainty_method is not None:
                    datapoint.row[unc_name] = un
            if args.individual_ensemble_predictions:
                for pred_name, model_preds in zip(task_names, ind_preds):
                    for idx, pred in enumerate(model_preds):
                        datapoint.row[pred_name + f"_model_{idx}"] = pred

        # Save
        with open(args.preds_path, 'w', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
            writer.writeheader()

            for datapoint in full_data:
                writer.writerow(datapoint.row)

        if evaluations is not None and args.evaluation_scores_path is not None:
            print(f"Saving uncertainty evaluations to {args.evaluation_scores_path}")
            if args.dataset_type == "multiclass":
                task_names = original_task_names
            with open(args.evaluation_scores_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["evaluation_method"] + task_names)
                for i, evaluation_method in enumerate(args.evaluation_methods):
                    writer.writerow([evaluation_method] + evaluations[i])

    if return_invalid_smiles:
        full_preds = []
        full_unc = []
        for full_index in range(len(full_data)):
            valid_index = full_to_valid_indices.get(full_index, None)
            if valid_index is not None:
                pred = preds[valid_index]
                un = unc[valid_index]
            else:
                pred = ["Invalid SMILES"] * num_tasks
                un = ["Invalid SMILES"] * num_unc_tasks
            full_preds.append(pred)
            full_unc.append(un)
        return full_preds, full_unc
    else:
        return preds, unc


@timeit()
def make_predictions(
    args: PredictArgs,
    smiles: List[List[str]] = None,
    model_objects: Tuple[
        PredictArgs,
        TrainArgs,
        List[MoleculeModel],
        List[Union[StandardScaler, AtomBondScaler]],
        int,
        List[str],
    ] = None,
    calibrator: UncertaintyCalibrator = None,
    return_invalid_smiles: bool = True,
    return_index_dict: bool = False,
    return_uncertainty: bool = False,
) -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :param model_objects: Tuple of output of load_model function which can be called separately outside this function. Preloaded model objects should have
                used the non-generator option for load_model if the objects are to be used multiple times or are intended to be used for calibration as well.
    :param calibrator: A :class: `~chemprop.uncertainty.UncertaintyCalibrator` object, for use in calibrating uncertainty predictions.
                Can be preloaded and provided as a function input or constructed within the function from arguments. The models and scalers used
                to initiate the calibrator must be lists instead of generators if the same calibrator is to be used multiple times or
                if the same models and scalers objects are also part of the provided model_objects input.
    :param return_invalid_smiles: Whether to return predictions of "Invalid SMILES" for invalid SMILES, otherwise will skip them in returned predictions.
    :param return_index_dict: Whether to return the prediction results as a dictionary keyed from the initial data indexes.
    :param return_uncertainty: Whether to return uncertainty predictions alongside the model value predictions.
    :return: A list of lists of target predictions. If returning uncertainty, a tuple containing first prediction values then uncertainty estimates.
    """
    if model_objects:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = model_objects
    else:
        (
            args,
            train_args,
            models,
            scalers,
            num_tasks,
            task_names,
        ) = load_model(args, generator=True)

    num_models = len(args.checkpoint_paths)

    set_features(args, train_args)

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(
        args, smiles
    )

    if args.uncertainty_method is None and (args.calibration_method is not None or args.evaluation_methods is not None):
        if args.dataset_type in ['classification', 'multiclass']:
            args.uncertainty_method = 'classification'
        else:
            raise ValueError('Cannot calibrate or evaluate uncertainty without selection of an uncertainty method.')


    if calibrator is None and args.calibration_path is not None:

        calibration_data = get_data(
            path=args.calibration_path,
            smiles_columns=args.smiles_columns,
            target_columns=task_names,
            args=args,
            features_path=args.calibration_features_path,
            features_generator=args.features_generator,
            phase_features_path=args.calibration_phase_features_path,
            atom_descriptors_path=args.calibration_atom_descriptors_path,
            bond_descriptors_path=args.calibration_bond_descriptors_path,
            max_data_size=args.max_data_size,
            loss_function=args.loss_function,
        )

        calibration_data_loader = MoleculeDataLoader(
            dataset=calibration_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if isinstance(models, List) and isinstance(scalers, List):
            calibration_models = models
            calibration_scalers = scalers
        else:
            calibration_model_objects = load_model(args, generator=True)
            calibration_models = calibration_model_objects[2]
            calibration_scalers = calibration_model_objects[3]

        calibrator = build_uncertainty_calibrator(
            calibration_method=args.calibration_method,
            uncertainty_method=args.uncertainty_method,
            interval_percentile=args.calibration_interval_percentile,
            regression_calibrator_metric=args.regression_calibrator_metric,
            calibration_data=calibration_data,
            calibration_data_loader=calibration_data_loader,
            models=calibration_models,
            scalers=calibration_scalers,
            num_models=num_models,
            dataset_type=args.dataset_type,
            loss_function=args.loss_function,
            uncertainty_dropout_p=args.uncertainty_dropout_p,
            dropout_sampling_size=args.dropout_sampling_size,
            spectra_phase_mask=getattr(train_args, "spectra_phase_mask", None),
        )

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        preds = [None] * len(full_data)
        unc = [None] * len(full_data)
    else:
        preds, unc = predict_and_save(
            args=args,
            train_args=train_args,
            test_data=test_data,
            task_names=task_names,
            num_tasks=num_tasks,
            test_data_loader=test_data_loader,
            full_data=full_data,
            full_to_valid_indices=full_to_valid_indices,
            models=models,
            scalers=scalers,
            num_models=num_models,
            calibrator=calibrator,
            return_invalid_smiles=return_invalid_smiles,
        )

    if return_index_dict:
        preds_dict = {}
        unc_dict = {}
        for i in range(len(full_data)):
            if return_invalid_smiles:
                preds_dict[i] = preds[i]
                unc_dict[i] = unc[i]
            else:
                valid_index = full_to_valid_indices.get(i, None)
                if valid_index is not None:
                    preds_dict[i] = preds[valid_index]
                    unc_dict[i] = unc[valid_index]
        if return_uncertainty:
            return preds_dict, unc_dict
        else:
            return preds_dict
    else:
        if return_uncertainty:
            return preds, unc
        else:
            return preds


def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())
