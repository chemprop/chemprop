from argparse import Namespace
import csv
from datetime import timedelta
from functools import wraps
import logging
import os
import pickle
import re
from time import time
from typing import Any, Callable, List, Tuple
import collections

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import PredictArgs, TrainArgs, FingerprintArgs
from chemprop.data import StandardScaler, AtomBondScaler, MoleculeDataset, preprocess_smiles_columns, get_task_names
from chemprop.models import MoleculeModel
from chemprop.nn_utils import NoamLR
from chemprop.models.ffn import MultiReadout


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def save_checkpoint(
    path: str,
    model: MoleculeModel,
    scaler: StandardScaler = None,
    features_scaler: StandardScaler = None,
    atom_descriptor_scaler: StandardScaler = None,
    bond_descriptor_scaler: StandardScaler = None,
    atom_bond_scaler: AtomBondScaler = None,
    args: TrainArgs = None,
) -> None:
    """
    Saves a model checkpoint.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the data.
    :param features_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the features.
    :param atom_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the atom descriptors.
    :param bond_descriptor_scaler: A :class:`~chemprop.data.scaler.StandardScaler` fitted on the bond descriptors.
    :param atom_bond_scaler: A :class:`~chemprop.data.scaler.AtomBondScaler` fitted on the atomic/bond targets.
    :param args: The :class:`~chemprop.args.TrainArgs` object containing the arguments the model was trained with.
    :param path: Path where checkpoint will be saved.
    """
    # Convert args to namespace for backwards compatibility
    if args is not None:
        args = Namespace(**args.as_dict())

    data_scaler = {"means": scaler.means, "stds": scaler.stds} if scaler is not None else None
    if atom_bond_scaler is not None:
        atom_bond_scaler = {"means": atom_bond_scaler.means, "stds": atom_bond_scaler.stds}
    if features_scaler is not None:
        features_scaler = {"means": features_scaler.means, "stds": features_scaler.stds}
    if atom_descriptor_scaler is not None:
        atom_descriptor_scaler = {
            "means": atom_descriptor_scaler.means,
            "stds": atom_descriptor_scaler.stds,
        }
    if bond_descriptor_scaler is not None:
        bond_descriptor_scaler = {"means": bond_descriptor_scaler.means, "stds": bond_descriptor_scaler.stds}

    state = {
        "args": args,
        "state_dict": model.state_dict(),
        "data_scaler": data_scaler,
        "features_scaler": features_scaler,
        "atom_descriptor_scaler": atom_descriptor_scaler,
        "bond_descriptor_scaler": bond_descriptor_scaler,
        "atom_bond_scaler": atom_bond_scaler,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str, device: torch.device = None, logger: logging.Logger = None
) -> MoleculeModel:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param device: Device where the model will be moved.
    :param logger: A logger for recording output.
    :return: The loaded :class:`~chemprop.models.model.MoleculeModel`.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args = TrainArgs()
    args.from_dict(vars(state["args"]), skip_unsettable=True)
    loaded_state_dict = state["state_dict"]

    if device is not None:
        args.device = device

    # Build model
    model = MoleculeModel(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for loaded_param_name in loaded_state_dict.keys():
        # Backward compatibility for parameter names
        if re.match(r"(encoder\.encoder\.)([Wc])", loaded_param_name) and not args.reaction_solvent:
            param_name = loaded_param_name.replace("encoder.encoder", "encoder.encoder.0")
        elif re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
        else:
            param_name = loaded_param_name

        # Load pretrained parameter, skipping unmatched parameters
        if param_name not in model_state_dict:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" cannot be found in model parameters.'
            )
        elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
            info(
                f'Warning: Pretrained parameter "{loaded_param_name}" '
                f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
                f"model parameter of shape {model_state_dict[param_name].shape}."
            )
        else:
            debug(f'Loading pretrained parameter "{loaded_param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.cuda:
        debug("Moving model to cuda")
    model = model.to(args.device)

    return model


def overwrite_state_dict(
    loaded_param_name: str,
    model_param_name: str,
    loaded_state_dict: collections.OrderedDict,
    model_state_dict: collections.OrderedDict,
    logger: logging.Logger = None,
) -> collections.OrderedDict:
    """
    Overwrites a given parameter in the current model with the loaded model.
    :param loaded_param_name: name of parameter in checkpoint model.
    :param model_param_name: name of parameter in current model.
    :param loaded_state_dict: state_dict for checkpoint model.
    :param model_state_dict: state_dict for current model.
    :param logger: A logger.
    :return: The updated state_dict for the current model.
    """
    debug = logger.debug if logger is not None else print

    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')

    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(
            f'Pretrained parameter "{loaded_param_name}" '
            f"of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding "
            f"model parameter of shape {model_state_dict[model_param_name].shape}."
        )

    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]

    return model_state_dict


def load_frzn_model(
    model: torch.nn,
    path: str,
    current_args: Namespace = None,
    cuda: bool = None,
    logger: logging.Logger = None,
) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model["state_dict"]
    loaded_args = loaded_mpnn_model["args"]

    # Backward compatibility for parameter names
    loaded_state_dict_keys = list(loaded_state_dict.keys())
    for loaded_param_name in loaded_state_dict_keys:
        if re.match(r"(^ffn)", loaded_param_name):
            param_name = loaded_param_name.replace("ffn", "readout")
            loaded_state_dict[param_name] = loaded_state_dict.pop(loaded_param_name)

    model_state_dict = model.state_dict()

    if loaded_args.number_of_molecules == 1 and current_args.number_of_molecules == 1:
        encoder_param_names = [
            "encoder.encoder.0.W_i.weight",
            "encoder.encoder.0.W_h.weight",
            "encoder.encoder.0.W_o.weight",
            "encoder.encoder.0.W_o.bias",
            "encoder.encoder.0.W_o_b.weight",
            "encoder.encoder.0.W_o_b.bias",
        ]
        if current_args.checkpoint_frzn is not None:
            # Freeze the MPNN
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            if isinstance(model.readout, nn.Sequential):  # Molecule properties
                ffn_param_names = [
                    [f"readout.{i*3+1}.weight", f"readout.{i*3+1}.bias"]
                    for i in range(current_args.frzn_ffn_layers)
                ]
            elif isinstance(model.readout, MultiReadout):  # Atomic/bond properties
                if model.readout.shared_ffn:
                    ffn_param_names = [
                        [f"readout.atom_ffn_base.0.{i*3+1}.weight", f"readout.atom_ffn_base.0.{i*3+1}.bias",
                        f"readout.bond_ffn_base.0.{i*3+1}.weight", f"readout.bond_ffn_base.0.{i*3+1}.bias"]
                        for i in range(current_args.frzn_ffn_layers)
                    ]
                else:
                    ffn_param_names = []
                    nmodels = len(model.readout.ffn_list)
                    for i in range(nmodels):
                        readout = model.readout.ffn_list[i]
                        if readout.constraint:
                            ffn_param_names.extend([
                                [f"readout.ffn_list.{i}.ffn.0.{j*3+1}.weight", f"readout.ffn_list.{i}.ffn.0.{j*3+1}.bias"]
                                for j in range(current_args.frzn_ffn_layers)
                            ])
                        else:
                            ffn_param_names.extend([
                                [f"readout.ffn_list.{i}.ffn_readout.{j*3+1}.weight", f"readout.ffn_list.{i}.ffn_readout.{j*3+1}.bias"]
                                for j in range(current_args.frzn_ffn_layers)
                            ])
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            # Freeze MPNN and FFN layers
            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.freeze_first_only:
            debug(
                "WARNING: --freeze_first_only flag cannot be used with number_of_molecules=1 (flag is ignored)"
            )

    elif loaded_args.number_of_molecules == 1 and current_args.number_of_molecules > 1:
        # TODO(degraff): these two `if`-blocks can be condensed into one
        if (
            current_args.checkpoint_frzn is not None
            and current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Only freeze first MPNN
            encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ]
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )
        if (
            current_args.checkpoint_frzn is not None
            and not current_args.freeze_first_only
            and current_args.frzn_ffn_layers <= 0
        ):  # Duplicate encoder from frozen checkpoint and overwrite all encoders
            loaded_encoder_param_names = [
                "encoder.encoder.0.W_i.weight",
                "encoder.encoder.0.W_h.weight",
                "encoder.encoder.0.W_o.weight",
                "encoder.encoder.0.W_o.bias",
            ] * current_args.number_of_molecules

            model_encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            model_encoder_param_names = [
                item for sublist in model_encoder_param_names for item in sublist
            ]

            for loaded_param_name, model_param_name in zip(
                loaded_encoder_param_names, model_encoder_param_names
            ):
                model_state_dict = overwrite_state_dict(
                    loaded_param_name, model_param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            raise ValueError(
                f"Number of molecules from checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must equal current number of molecules ({current_args.number_of_molecules})!"
            )

    elif loaded_args.number_of_molecules > 1 and current_args.number_of_molecules > 1:
        if loaded_args.number_of_molecules != current_args.number_of_molecules:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                f"must either match current model ({current_args.number_of_molecules}) or equal 1."
            )

        if current_args.freeze_first_only:
            raise ValueError(
                f"Number of molecules in checkpoint_frzn ({loaded_args.number_of_molecules}) "
                "must be equal to 1 for freeze_first_only to be used!"
            )

        if (current_args.checkpoint_frzn is not None) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]

            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers > 0:
            encoder_param_names = [
                [
                    (
                        f"encoder.encoder.{mol_num}.W_i.weight",
                        f"encoder.encoder.{mol_num}.W_h.weight",
                        f"encoder.encoder.{mol_num}.W_o.weight",
                        f"encoder.encoder.{mol_num}.W_o.bias",
                    )
                ]
                for mol_num in range(current_args.number_of_molecules)
            ]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]
            ffn_param_names = [
                [f"readout.{i*3+1}.weight", f"readout.{i*3+1}.bias"]
                for i in range(current_args.frzn_ffn_layers)
            ]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]

            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(
                    param_name, param_name, loaded_state_dict, model_state_dict
                )

        if current_args.frzn_ffn_layers >= current_args.ffn_num_layers:
            raise ValueError(
                f"Number of frozen FFN layers ({current_args.frzn_ffn_layers}) "
                f"must be less than the number of FFN layers ({current_args.ffn_num_layers})!"
            )

    # Load pretrained weights
    model.load_state_dict(model_state_dict)

    return model


def load_scalers(
    path: str,
) -> Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler, List[StandardScaler]]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    if state["data_scaler"] is not None:
        scaler = StandardScaler(state["data_scaler"]["means"], state["data_scaler"]["stds"])
    else:
        scaler = None

    if state["features_scaler"] is not None:
        features_scaler = StandardScaler(
            state["features_scaler"]["means"], state["features_scaler"]["stds"], replace_nan_token=0
        )
    else:
        features_scaler = None

    if "atom_descriptor_scaler" in state.keys() and state["atom_descriptor_scaler"] is not None:
        atom_descriptor_scaler = StandardScaler(
            state["atom_descriptor_scaler"]["means"],
            state["atom_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        atom_descriptor_scaler = None

    if "bond_descriptor_scaler" in state.keys() and state["bond_descriptor_scaler"] is not None:
        bond_descriptor_scaler = StandardScaler(
            state["bond_descriptor_scaler"]["means"],
            state["bond_descriptor_scaler"]["stds"],
            replace_nan_token=0,
        )
    else:
        bond_descriptor_scaler = None

    if "atom_bond_scaler" in state.keys() and state["atom_bond_scaler"] is not None:
        atom_bond_scaler =AtomBondScaler(
            state["atom_bond_scaler"]["means"],
            state["atom_bond_scaler"]["stds"],
            replace_nan_token=0,
            n_atom_targets=len(state["args"].atom_targets),
            n_bond_targets=len(state["args"].bond_targets),
        )
    else:
        atom_bond_scaler = None

    return scaler, features_scaler, atom_descriptor_scaler, bond_descriptor_scaler, atom_bond_scaler


def load_args(path: str) -> TrainArgs:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The :class:`~chemprop.args.TrainArgs` object that the model was trained with.
    """
    args = TrainArgs()
    args.from_dict(
        vars(torch.load(path, map_location=lambda storage, loc: storage)["args"]),
        skip_unsettable=True,
    )

    return args


def load_task_names(path: str) -> List[str]:
    """
    Loads the task names a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A list of the task names that the model was trained with.
    """
    return load_args(path).task_names


def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)


def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None
) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, "verbose.log"))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, "quiet.log"))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """

    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """

        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f"Elapsed time = {delta}")

            return result

        return wrap

    return timeit_decorator


def save_smiles_splits(
    data_path: str,
    save_dir: str,
    task_names: List[str] = None,
    features_path: List[str] = None,
    constraints_path: str = None,
    train_data: MoleculeDataset = None,
    val_data: MoleculeDataset = None,
    test_data: MoleculeDataset = None,
    smiles_columns: List[str] = None,
    loss_function: str = None,
    logger: logging.Logger = None,
) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param constraints_path: Path to constraints applied to atomic/bond properties prediction.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param loss_function: The loss function to be used in training.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)

    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        f = open(data_path)
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info(
                    "Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated."
                )
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(
            path=data_path,
            smiles_columns=smiles_columns,
            loss_function=loss_function,
            )

    if loss_function == "quantile_interval":
        num_tasks = len(task_names) // 2
        task_names = task_names[:num_tasks]

    features_header = []
    if features_path is not None:
        extension_sets = set([os.path.splitext(feat_path)[1] for feat_path in features_path])
        if extension_sets == {'.csv'}:
            for feat_path in features_path:
                with open(feat_path, "r") as f:
                    reader = csv.reader(f)
                    feat_header = next(reader)
                    features_header.extend(feat_header)

    if constraints_path is not None:
        with open(constraints_path, "r") as f:
            reader = csv.reader(f)
            constraints_header = next(reader)

    all_split_indices = []
    for dataset, name in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f"{name}_smiles.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            if smiles_columns[0] == "":
                writer.writerow(["smiles"])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f"{name}_full.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                targets = [x.tolist() if isinstance(x, np.ndarray) else x for x in dataset_targets[i]]
                # correct the number of targets when running quantile regression
                targets = targets[:len(task_names)]
                writer.writerow(smiles + targets)

        if features_path is not None:
            dataset_features = dataset.features()
            if extension_sets == {'.csv'}:
                with open(os.path.join(save_dir, f"{name}_features.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(features_header)
                    writer.writerows(dataset_features)
            else:
                np.save(os.path.join(save_dir, f"{name}_features.npy"), dataset_features)

        if constraints_path is not None:
            dataset_constraints = [d.raw_constraints for d in dataset._data]
            with open(os.path.join(save_dir, f"{name}_constraints.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(constraints_header)
                writer.writerows(dataset_constraints)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(
                        f"Warning: SMILES string in {name} could not be found in data file, and "
                        "likely came from a secondary data file. The pickle file of split indices "
                        "can only indicate indices for a single file and will not be generated."
                    )
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == "train":
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f"{name}_weights.csv"), "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["data weights"])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, "split_indices.pckl"), "wb") as f:
            pickle.dump(all_split_indices, f)


def update_prediction_args(
    predict_args: PredictArgs,
    train_args: TrainArgs,
    missing_to_defaults: bool = True,
    validate_feature_sources: bool = True,
) -> None:
    """
    Updates prediction arguments with training arguments loaded from a checkpoint file.
    If an argument is present in both, the prediction argument will be used.

    Also raises errors for situations where the prediction arguments and training arguments
    are different but must match for proper function.

    :param predict_args: The :class:`~chemprop.args.PredictArgs` object containing the arguments to use for making predictions.
    :param train_args: The :class:`~chemprop.args.TrainArgs` object containing the arguments used to train the model previously.
    :param missing_to_defaults: Whether to replace missing training arguments with the current defaults for :class: `~chemprop.args.TrainArgs`.
        This is used for backwards compatibility.
    :param validate_feature_sources: Indicates whether the feature sources (from path or generator) are checked for consistency between
        the training and prediction arguments. This is not necessary for fingerprint generation, where molecule features are not used.
    """
    for key, value in vars(train_args).items():
        if not hasattr(predict_args, key):
            setattr(predict_args, key, value)

    if missing_to_defaults:
        # If a default argument would cause different behavior than occurred in legacy checkpoints before the argument existed,
        # then that argument must be included in the `override_defaults` dictionary to force the legacy behavior.
        override_defaults = {
            "bond_descriptors_scaling": False,
            "no_bond_descriptors_scaling": True,
            "atom_descriptors_scaling": False,
            "no_atom_descriptors_scaling": True,
        }
        default_train_args = TrainArgs().parse_args(
            ["--data_path", None, "--dataset_type", str(train_args.dataset_type)]
        )
        for key, value in vars(default_train_args).items():
            if not hasattr(predict_args, key):
                setattr(predict_args, key, override_defaults.get(key, value))

    # Same number of molecules must be used in training as in making predictions
    if train_args.number_of_molecules != predict_args.number_of_molecules and not (
        isinstance(predict_args, FingerprintArgs)
        and predict_args.fingerprint_type == "MPN"
        and predict_args.mpn_shared
        and predict_args.number_of_molecules == 1
    ):
        raise ValueError(
            "A different number of molecules was used in training "
            "model than is specified for prediction. This is only supported for models with shared MPN networks"
            f"and a fingerprint type of MPN. {train_args.number_of_molecules} smiles fields must be provided."
        )

    # if atom or bond features were scaled, the same must be done during prediction
    if train_args.features_scaling != predict_args.features_scaling:
        raise ValueError(
            "If scaling of the additional features was done during training, the "
            "same must be done during prediction."
        )

    # If atom descriptors were used during training, they must be used when predicting and vice-versa
    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError(
            "The use of atom descriptors is inconsistent between training and prediction. "
            "If atom descriptors were used during training, they must be specified again "
            "during prediction using the same type of descriptors as before. "
            "If they were not used during training, they cannot be specified during prediction."
        )

    # If bond features were used during training, they must be used when predicting and vice-versa
    if train_args.bond_descriptors != predict_args.bond_descriptors:
        raise ValueError(
            "The use of bond descriptors is inconsistent between training and prediction. "
            "If bond descriptors were used during training, they must be specified again "
            "during prediction using the same type of descriptors as before. "
            "If they were not used during training, they cannot be specified during prediction."
        )

    # If constraints were used during training, they must be used when predicting and vice-versa
    if (train_args.constraints_path is None) != (predict_args.constraints_path is None):
        raise ValueError(
            "The use of constraints is different between training and prediction. If you applied constraints "
            "for training, please specify a path to new constraints for prediction."
        )

    # If features were used during training, they must be used when predicting
    if validate_feature_sources:
        if ((train_args.features_path is None) != (predict_args.features_path is None)) or (
            (train_args.features_generator is None) != (predict_args.features_generator is None)
        ):
            raise ValueError(
                "Features were used during training so they must be specified again during "
                "prediction using the same type of features as before "
                "(with either --features_generator or --features_path "
                "and using --no_features_scaling if applicable)."
            )


def multitask_mean(
    scores: np.ndarray,
    metric: str,
    axis: int = None,
    ignore_nan_metrics: bool = False,
) -> float:
    """
    A function for combining the metric scores across different
    model tasks into a single score. When the metric being used
    is one that varies with the magnitude of the task (such as RMSE),
    a geometric mean is used, otherwise a more typical arithmetic mean
    is used. This prevents a task with a larger magnitude from dominating
    over one with a smaller magnitude (e.g., temperature and pressure).

    :param scores: The scores from different tasks for a single metric.
    :param metric: The metric used to generate the scores.
    :param axis: The axis along which to take the mean.
    :param ignore_nan_metrics: Ignore invalid task metrics (NaNs) when computing average metrics across tasks.
    :return: The combined score across the tasks.
    """
    scale_dependent_metrics = ["rmse", "mae", "mse", "bounded_rmse", "bounded_mae", "bounded_mse", "quantile"]
    nonscale_dependent_metrics = [
        "auc", "prc-auc", "r2", "accuracy", "cross_entropy",
        "binary_cross_entropy", "sid", "wasserstein", "f1", "mcc", "recall", "precision", "balanced_accuracy", "confusion_matrix"

    ]

    mean_fn = np.nanmean if ignore_nan_metrics else np.mean

    if metric in scale_dependent_metrics:
        return np.exp(mean_fn(np.log(scores), axis=axis))
    elif metric in nonscale_dependent_metrics:
        return mean_fn(scores, axis=axis)
    else:
        raise NotImplementedError(
            f"The metric used, {metric}, has not been added to the list of\
                metrics that are scale-dependent or not scale-dependent.\
                This metric must be added to the appropriate list in the multitask_mean\
                function in `chemprop/utils.py` in order to be used."
        )
