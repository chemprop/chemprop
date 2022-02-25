from typing import List, Tuple, Optional
import csv
from collections import OrderedDict

import numpy as np

from chemprop.models import MoleculeModel
from chemprop.args import UncertaintyArgs, TrainArgs
from chemprop.data import StandardScaler, get_data
from chemprop.utils import timeit, makedirs
from chemprop.train import load_model, set_features, load_data, predict_and_save
from .uncertainty_calibrator import UncertaintyCalibrator, uncertainty_calibrator_builder
from .uncertainty_estimator import UncertaintyEstimator


@timeit()
def run_uncertainty(args: UncertaintyArgs,
                    smiles: List[List[str]] = None,
                    model_objects: Tuple[UncertaintyArgs, TrainArgs, List[MoleculeModel], List[StandardScaler], int, List[str]] = None,
                    calibrator: UncertaintyCalibrator = None,
                    return_invalid_smiles: bool = True,
                    return_index_dict: bool = False,
                    save_results: bool = True,
                    ) -> List[List[Optional[float]]]:
    """
    return predictions and uncertainty metric
    """
    if model_objects:
        args, train_args, models, scalers, num_tasks, task_names = model_objects
    else:
        args, train_args, models, scalers, num_tasks, task_names = load_model(args, generator=True)

    set_features(args, train_args)

    # Build the calibrator object if calibration is being carried out
    if calibrator is None and args.calibration_path is not None:

        calibration_data = get_data(
            path=args.calibration_path,
            smiles_columns=args.smiles_columns,
            target_columns=task_names,
            features_path=args.calibration_features_path,
            features_generator=args.features_generator,
            phase_features_path=args.calibration_phase_features_path,
            atom_descriptors_path=args.calibration_atom_descriptors_path,
            bond_features_path=args.calibration_bond_features_path,
            max_data_size=args.max_data_size,
            loss_function=args.loss_function,
        )

        calibrator = uncertainty_calibrator_builder(
            calibration_method=args.calibration_method,
            uncertainty_method=args.uncertainty_method,
            calibration_data=calibration_data,
            calibration_metric=args.calibration_metric,
            models=models,
            scalers=scalers,
            dataset_type=args.dataset_type,
            loss_function=args.loss_function,
        )

    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, _, full_to_valid_indices = load_data(args, smiles)
    
    estimator = UncertaintyEstimator(
        test_data=test_data,
        models=models,
        scalers=scalers,
        dataset_type=args.dataset_type,
        loss_function=args.loss_function,
    )

    preds, unc = estimator.calculate_uncertainty(calibrator=calibrator) # preds and unc are lists of shape(data,tasks)

    # Save results
    if save_results:
        print(f'Saving predictions to {args.preds_path}')
        assert len(test_data) == len(preds)
        assert len(test_data) == len(unc)

        makedirs(args.preds_path, isfile=True)

        # Set multiclass column names, update num_tasks definition for multiclass
        if args.dataset_type == 'multiclass':
            task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
            num_tasks = num_tasks * args.multiclass_num_classes

        # Copy predictions over to full_data
        for full_index, datapoint in enumerate(full_data):
            valid_index = full_to_valid_indices.get(full_index, None)
            d_preds = preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * num_tasks
            d_unc = unc[valid_index] if valid_index is not None else ['Invalid SMILES'] * num_tasks

            # Reshape multiclass to merge task and class dimension, with updated num_tasks
            if args.dataset_type == 'multiclass':
                if isinstance(d_preds, np.ndarray) and d_preds.ndim > 1:
                    d_preds = d_preds.reshape((num_tasks))
                    d_unc = d_unc.reshape((num_tasks))

            # If extra columns have been dropped, add back in SMILES columns
            if args.drop_extra_columns:
                datapoint.row = OrderedDict()

                smiles_columns = args.smiles_columns

                for column, smiles in zip(smiles_columns, datapoint.smiles):
                    datapoint.row[column] = smiles

            # Add predictions columns
            if calibrator == None: args.calibration_metric = 'uncal'
            unc_names = [name + f'_{args.calibration_metric}_{args.uncertainty_method}' for name in task_names]
            for pred_name, unc_name, pred, un in zip(task_names, unc_names, d_preds, d_unc):
                datapoint.row[pred_name] = pred
                datapoint.row[unc_name] = un

        # Save
        with open(args.preds_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
            writer.writeheader()
            for datapoint in full_data:
                writer.writerow(datapoint.row)
    
    # Report results
    if return_index_dict:
        preds_dict = {}
        for i in range(len(full_data)):
            valid_index = full_to_valid_indices.get(i, None)
            if valid_index is None:
                if return_invalid_smiles:
                    preds_dict[i] = (['Invalid SMILES'] * num_tasks, ['Invalid SMILES'] * num_tasks)
            else:
                preds_dict[i] = (preds[valid_index], unc[valid_index])
        return preds_dict
    else:
        preds_list = []
        for i in range(len(full_data)):
            valid_index = full_to_valid_indices.get(i, None)
            if valid_index is None:
                if return_invalid_smiles:
                    preds_list.append((['Invalid SMILES'] * num_tasks, ['Invalid SMILES'] * num_tasks))
            else:
                preds_list.append((preds[valid_index], unc[valid_index]))
        return preds_list


def chemprop_uncertainty() -> None:
    """

    """
    run_uncertainty(args=UncertaintyArgs().parse_args())
