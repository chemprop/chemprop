from typing import List, Tuple, Optional, Union

import numpy as np

from chemprop.models import MoleculeModel
from chemprop.args import PredictArgs, UncertaintyArgs, TrainArgs
from chemprop.data import StandardScaler, get_data
from chemprop.utils import timeit
from chemprop.train import load_model, set_features, load_data, predict_and_save
from .uncertainty_calibrator import UncertaintyCalibrator, uncertainty_calibrator_builder


@timeit()
def run_uncertainty(args: UncertaintyArgs,
                    smiles: List[List[str]] = None,
                    model_objects: Tuple[UncertaintyArgs, TrainArgs, List[MoleculeModel], List[StandardScaler], int, List[str]] = None,
                    calibrator: UncertaintyCalibrator = None,
                    return_invalid_smiles: bool = True,
                    return_index_dict: bool = False) -> List[List[Optional[float]]]:
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
        )


    # Note: to get the invalid SMILES for your data, use the get_invalid_smiles_from_file or get_invalid_smiles_from_list functions from data/utils.py
    full_data, test_data, test_data_loader, full_to_valid_indices = load_data(args, smiles)
    
    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        avg_preds = [None] * len(full_data)
    else:
        avg_preds = predict_and_save(
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
            return_invalid_smiles=return_invalid_smiles,
        )
    
    if return_index_dict:
        preds_dict = {}
        for i in range(len(full_data)):
            if return_invalid_smiles:
                preds_dict[i] = avg_preds[i]
            else:
                valid_index = full_to_valid_indices.get(i, None)
                if valid_index is not None:
                    preds_dict[i] = avg_preds[valid_index]
        return preds_dict
    else:
        return avg_preds


def chemprop_uncertainty() -> None:
    """

    """
    run_uncertainty(args=UncertaintyArgs().parse_args())
