from chemprop.v2.args import PredictArgs
from typing import Union, Optional
import torch
from chemprop.v2.models import modules, models
from chemprop.v2.data import get_data, get_data_from_smiles, MolGraphDataLoader, MoleculeDataset
from chemprop.v2.models import MPNN, ClassificationMPNN, DirichletClassificationMPNN, MulticlassMPNN, DirichletMulticlassMPNN, RegressionMPNN, MveRegressionMPNN, SpectralMPNN 
from chemprop.v2.uncertainty import UncertaintyEstimator
# build_uncertainty_evaluator should be imported from chemprop.v2.uncertainty but it isn't implemented yet: to do
def build_uncertainty_evaluator():
    pass
from chemprop.v2.utils.utils import timeit
from chemprop.v2.utils.utils_old import makedirs
import numpy as np
from collections import OrderedDict
import csv

# This function is missing in my version. Not sure what it does so implement later (to do)
# from chemprop.multitask_utils import reshape_values
def reshape_values():
    pass

from logging import Logger

def load_model(checkpoint_paths: str, 
               # (Not yet implemented) generator: bool = False,
               ):
    """
    Function to load a model or ensemble of models from file. If generator is True, a generator of the respective model and scaler 
    objects is returned (memory efficient), else the full list (holding all models in memory, necessary for preloading).

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param generator: A boolean to return a generator instead of a list of models and scalers.
    :return: A tuple of updated prediction arguments, training arguments, a list or generator object of models, a list or 
                 generator object of scalers, the number of tasks and their respective names.
    """
    # Remove assert statement after support for mulitple models is implemented
    assert len(checkpoint_paths) == 1
    checkpoint_path = checkpoint_paths[0]
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda:0'))

    mpn_block = modules.molecule_block()
    try:
        exec('global models_list; models_list = models.' + checkpoint['model_type'] + '.load_from_checkpoint("' + checkpoint_path + '", mpn_block = mpn_block,  map_location=torch.device("cpu"))')
    except:
        exec('global models_list; models_list = models.' + checkpoint['model_type'] + '.load_from_checkpoint("' + checkpoint_path + '", mpn_block = mpn_block,  map_location=torch.device("cuda:0"))')

    return checkpoint, [models_list]

def load_data(smiles: list[list[str]] = None,
              path: str = None,
              smiles_columns: Union[str, list[str]] = None,
              target_columns: list[str] = None,
              ignore_columns: list[str] = None,
              skip_invalid_smiles: bool = True,
              data_weights_path: str = None,
              features_path: list[str] = None,
              features_generator: list[str] = None,
              phase_features_path: str = None,
              atom_descriptors_path: str = None,
              bond_features_path: str = None,
              max_data_size: int = None,
              store_row: bool = False,
              logger: Logger = None,
              loss_function: str = None,
              skip_none_targets: bool = False,
              atom_descriptors_type: str = None,
              batch_size: int = None,
              num_workers: int = None,
              ):
    """
    Function to load data from a list of smiles or a file.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: A list of list of smiles, or None if data is to be read from file
    :return: A tuple of a :class:`~chemprop.data.MoleculeDataset` containing all datapoints and a :class:`~chemprop.data.MoleculeDataLoader`.
    """
    print("Loading data")
    if smiles is not None:
        data_set = get_data_from_smiles(
            smiles = smiles,
            skip_invalid_smiles = False,
            logger = logger,
            features_generator=features_generator,
        )
    else:
        data_set = get_data(
            path = path,
            smiles_columns = smiles_columns,
            target_columns = target_columns,
            ignore_columns = ignore_columns,
            skip_invalid_smiles = skip_invalid_smiles,
            data_weights_path = data_weights_path,
            features_path = features_path,
            features_generator = features_generator,
            phase_features_path = phase_features_path,
            atom_descriptors_path = atom_descriptors_path,
            bond_features_path = bond_features_path,
            max_data_size = max_data_size,
            store_row = store_row,
            logger = logger,
            loss_function = loss_function,
            skip_none_targets = skip_none_targets,
            atom_descriptors_type = atom_descriptors_type,
        )

    data_loader = MolGraphDataLoader(
        dataset = data_set, batch_size = batch_size, num_workers = num_workers, shuffle=False
    )

    return data_set, data_loader

def set_features(*arguments):
    """
    Function to set extra options.

    :param args: A :class:`~chemprop.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param train_args: A :class:`~chemprop.args.TrainArgs` object containing arguments for training the model.
    """
    pass

def predict_and_save(
    test_data: MoleculeDataset,
    task_names: list[str],
    num_tasks: int,
    test_data_loader: MolGraphDataLoader,
    models_list: list[Union[ClassificationMPNN, DirichletClassificationMPNN, MulticlassMPNN, DirichletMulticlassMPNN, RegressionMPNN, MveRegressionMPNN, SpectralMPNN]],
    # (Not yet implemented) calibrator: UncertaintyCalibrator = None,
    save_results: bool = True,
    uncertainty_method: str = None,
    dataset_type: str = None,
    individual_ensemble_predictions: bool = None, 
    calibrator = None,
    is_atom_bond_targets = None,
    calibration_method = None,
    atom_targets = None, 
    bond_targets = None,
    evaluation_methods = None,
    path: str = None,
    smiles_columns: Union[str, list[str]] = None,
    target_columns: list[str] = None,
    ignore_columns: list[str] = None,
    skip_invalid_smiles: bool = True,
    data_weights_path: str = None,
    features_path: list[str] = None,
    features_generator: list[str] = None,
    phase_features_path: str = None,
    atom_descriptors_path: str = None,
    bond_features_path: str = None,
    max_data_size: int = None,
    store_row: bool = False,
    logger: Logger = None,
    loss_function: str = None,
    skip_none_targets: bool = False,
    atom_descriptors_type: str = None,
    multiclass_num_classes: int = None,
    drop_extra_columns: bool = None,
    evaluation_scores_path: str = None,
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
        test_data = test_data,
        test_data_loader = test_data_loader,
        models = models_list,
        dataset_type = dataset_type,
        individual_ensemble_predictions = individual_ensemble_predictions,
        spectra_phase_mask= None # getattr(train_args, "spectra_phase_mask", None),
    )

    preds, unc = estimator.calculate_uncertainty(
        # (Not yet implemented) calibrator = calibrator
    ) # preds and unc are lists of shape(data,tasks)

    if calibrator is not None and is_atom_bond_targets and calibration_method == "isotonic":
        unc = reshape_values(unc, test_data, len(atom_targets), len(bond_targets), num_tasks)

    if individual_ensemble_predictions:
        individual_preds = (
            estimator.individual_predictions()
        )  # shape(data, tasks, ensemble) or (data, tasks, classes, ensemble)

    if evaluation_methods is not None:

        evaluation_data = get_data(
            path = path,
            smiles_columns = smiles_columns,
            target_columns = target_columns,
            ignore_columns = ignore_columns,
            skip_invalid_smiles = skip_invalid_smiles,
            data_weights_path = data_weights_path,
            features_path = features_path,
            features_generator = features_generator,
            phase_features_path = phase_features_path,
            atom_descriptors_path = atom_descriptors_path,
            bond_features_path = bond_features_path,
            max_data_size = max_data_size,
            store_row = store_row,
            logger = logger,
            loss_function = loss_function,
            skip_none_targets = skip_none_targets,
            atom_descriptors_type = atom_descriptors_type,
        )

        evaluators = []
        for evaluation_method in evaluation_methods:
            evaluator = build_uncertainty_evaluator(
                evaluation_method=evaluation_method,
                calibration_method=calibration_method,
                uncertainty_method=uncertainty_method,
                dataset_type=dataset_type,
                loss_function=loss_function,
                calibrator=calibrator,
                is_atom_bond_targets=is_atom_bond_targets,
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
        print(f"Saving predictions to {path}")
        assert len(test_data) == len(preds)
        assert len(test_data) == len(unc)

        makedirs(path, isfile=True)

        # Set multiclass column names, update num_tasks definitions
        if dataset_type == "multiclass":
            original_task_names = task_names
            task_names = [
                f"{name}_class_{i}"
                for name in task_names
                for i in range(multiclass_num_classes)
            ]
            num_tasks = num_tasks * multiclass_num_classes
        if uncertainty_method == "spectra_roundrobin":
            num_unc_tasks = 1
        else:
            num_unc_tasks = num_tasks

        datapointrow = {}
        for full_index, datapoint in enumerate(test_data):
            row = {}
            d_preds = preds[full_index]
            d_unc = [unc[full_index]]
            # Reshape multiclass to merge task and class dimension, with updated num_tasks
            if dataset_type == "multiclass":
                d_preds = np.array(d_preds).reshape((num_tasks))
                d_unc = np.array(d_unc).reshape((num_unc_tasks))
                if individual_ensemble_predictions:
                    ind_preds = ind_preds.reshape(
                        (num_tasks, len(models_list))
                    )

            # If extra columns have been dropped, add back in SMILES columns
            if drop_extra_columns:
                row = OrderedDict()

                smiles_columns = smiles_columns

                for column, smiles in zip(smiles_columns, datapoint.smiles):
                    row[column] = smiles

            # Add predictions columns
            if uncertainty_method == "spectra_roundrobin":
                unc_names = [estimator.label]
            else:
                unc_names = [name + f"_{estimator.label}" for name in task_names]

            for pred_name, unc_name, pred, un in zip(
                task_names, unc_names, d_preds, d_unc
            ):
                row[pred_name] = pred

                if uncertainty_method is not None:
                    row[unc_name] = un
            if individual_ensemble_predictions:
                for pred_name, model_preds in zip(task_names, ind_preds):
                    for idx, pred in enumerate(model_preds):
                        row[pred_name + f"_model_{idx}"] = pred
            datapointrow[full_index] = row
        # Save
        save_path = path.split(".csv")[0] + "_withResults.csv"
        with open(path, 'w', newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()

            for i in range(len(datapointrow)):
                writer.writerow(datapointrow[i])

        if evaluations is not None and evaluation_scores_path is not None:
            print(f"Saving uncertainty evaluations to {evaluation_scores_path}")
            if dataset_type == "multiclass":
                task_names = original_task_names
            with open(evaluation_scores_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["evaluation_method"] + task_names)
                for i, evaluation_method in enumerate(evaluation_methods):
                    writer.writerow([evaluation_method] + evaluations[i])

    return preds, unc

@timeit()
def make_predictions(
    args: PredictArgs,
    smiles: list[list[str]] = None,
    # (Not yet implemented) calibrator: UncertaintyCalibrator = None,
    # (Not yet implemented) return_invalid_smiles: bool = True,
    # (Not yet implemented) return_index_dict: bool = False,
    return_uncertainty: bool = False,
) -> tuple[list[Optional[float]]]:
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

    checkpoint, models_list = load_model(
        checkpoint_paths = args.checkpoint_paths,
    )

    set_features() # Does nothing currently (figure out what it is supposed to do and implement it: to do)

    test_data, test_data_loader = load_data(
        smiles = smiles,
        path = args.test_path,
        smiles_columns = args.smiles_columns,
        target_columns = [],
        ignore_columns = [],
        skip_invalid_smiles = False,
        data_weights_path = None,
        features_path = args.features_path,
        features_generator = args.features_generator,
        phase_features_path = args.phase_features_path,
        atom_descriptors_path = args.atom_descriptors_path,
        # Note these aren't named the same
        bond_features_path = args.bond_descriptors_path,
        max_data_size = args.max_data_size,
        store_row = not args.drop_extra_columns,
        logger = None,
        # args doesn't have a loss function 
        loss_function = None, #args.loss_function,
        skip_none_targets = None,
        atom_descriptors_type = args.atom_descriptors,
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )

    # calibrator code goes here (add calibrator code: to do)

    # Currently only works with a single model (extend to multiple models: to do)
    # Only scaling targets is implemented (extend to scale features: to do)
    preds, unc = predict_and_save(
            test_data = test_data,
            num_tasks = checkpoint['hyper_parameters']['n_tasks'],
            # task_names was in the TrainArgs. We'll need to save it to the checkpoint if we really want it
            task_names=["Predict"],
            test_data_loader = test_data_loader,
            models_list = models_list,
            # (Not yet implemented) calibrator=calibrator,
            # (Not yet implemented) return_invalid_smiles = False,
            individual_ensemble_predictions = args.individual_ensemble_predictions,
            path = args.test_path,
            smiles_columns = args.smiles_columns,
            target_columns = [],
            ignore_columns = [],
            skip_invalid_smiles = False,
            data_weights_path = None,
            features_path = args.features_path,
            features_generator = args.features_generator,
            phase_features_path = args.phase_features_path,
            atom_descriptors_path = args.atom_descriptors_path,
            # Note that these aren't the same name. There was a PR that changed the name.
            bond_features_path = args.bond_descriptors_path,
            max_data_size = args.max_data_size,
            store_row = not args.drop_extra_columns,
            logger = None,
            # args has no loss_function
            loss_function = None, # args.loss_function,
            skip_none_targets = None,
            atom_descriptors_type = args.atom_descriptors,
            # args has no multiclass_num_classes
            multiclass_num_classes = None, # args.multiclass_num_classes,
            drop_extra_columns = args.drop_extra_columns,
            evaluation_scores_path = args.evaluation_scores_path,
        )

    if return_uncertainty:
        return preds, unc
    else:
        return preds

def chemprop_predict() -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`chemprop_predict`.
    """
    make_predictions(args=PredictArgs().parse_args())