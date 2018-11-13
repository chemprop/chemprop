import logging
import math
import os
from typing import Callable, Tuple, Union
from argparse import Namespace

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score
import torch
import torch.nn as nn

from chemprop.data import StandardScaler
from chemprop.models import build_model


def save_checkpoint(model: nn.Module,
                    scaler: StandardScaler,
                    features_scaler: StandardScaler,
                    args: Namespace,
                    path: str):
    """
    Saves a model checkpoint.

    :param model: A PyTorch model.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    if args.moe:
        state['domain_encs'] = model.get_domain_encs()
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = False,
                    num_tasks: int = None,
                    dataset_type: str = None,
                    encoder_only: bool = False,
                    logger: logging.Logger = None) -> Tuple[nn.Module, StandardScaler, StandardScaler, Namespace]:
    """
    Loads a model checkpoint and optionally the scaler the model was trained with.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments.
    :param cuda: Whether to move model to cuda.
    :param num_tasks: The number of tasks. Only necessary if different now than when trained.
    :param dataset_type: The type of the dataset ("classification" or "regression"). Only necessary
    if different now than when trained.
    :param encoder_only: Whether to only load weights from encoder.
    :param logger: A logger.
    :return: The loaded model, data scaler, features scaler, and loaded args.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    # Update args with current args
    args.cuda = cuda
    args.num_tasks = num_tasks or args.num_tasks
    args.dataset_type = dataset_type or args.dataset_type

    if current_args is not None:
        for key, value in vars(current_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)

    model = build_model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if encoder_only and 'encoder' not in param_name:
            continue

        if param_name not in model_state_dict:
            if logger is not None:
                logger.info('Pretrained parameter "{}" cannot be found in model parameters. Skipping.'.format(param_name))
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            if logger is not None:
                logger.info('Pretrained parameter "{}" of shape {} does not match corresponding '
                            'model parameter of shape {}.Skipping.'.format(param_name,
                                                                           loaded_state_dict[param_name].shape,
                                                                           model_state_dict[param_name].shape))
        else:
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if args.moe:
        domain_encs = state['domain_encs']
        if args.cuda:
            domain_encs = [encs.cuda() for encs in domain_encs]
        model.set_domain_encs(domain_encs)

    if cuda:
        print('Moving model to cuda')
        model = model.cuda()

    scaler = StandardScaler(state['scaler']['means'], state['scaler']['stds']) if state['scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'], state['features_scaler']['stds'], replace_nan_token=0) if state['features_scaler'] is not None else None

    return model, scaler, features_scaler, args


def get_loss_func(dataset_type: str) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param dataset_type: The dataset type ("classification" or "regression" or "regression_with_binning").
    :return: A PyTorch loss function.
    """
    if dataset_type == 'classification':
        return nn.BCELoss(reduction='none')
    
    if dataset_type == 'regression_with_binning':
        return nn.CrossEntropyLoss(reduction='none')

    if dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    raise ValueError('Dataset type "{}" not supported.'.format(dataset_type))


def get_metric_func(metric: str) -> Callable:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: The name of the metric.
    :return: A metric function which takes as arguments a list of targets and a list of predictions.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        def metric_func(targets, preds):
            precision, recall, _ = precision_recall_curve(targets, preds)
            return auc(recall, precision)
        return metric_func

    if metric == 'rmse':
        return lambda targets, preds: math.sqrt(mean_squared_error(targets, preds))

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score
    
    if metric == 'accuracy':
        def metric_func(targets, preds):
            hard_preds = [1 if p > 0.5 else 0 for p in preds]
            return accuracy_score(targets, hard_preds)
        return metric_func

    raise ValueError('Metric "{}" not supported.'.format(metric))


def set_logger(logger: logging.Logger, save_dir: str, quiet: bool):
    """
    Sets up a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param logger: A logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    """
    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)

    fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
    fh_v.setLevel(logging.DEBUG)
    fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
    fh_q.setLevel(logging.INFO)

    logger.addHandler(ch)
    logger.addHandler(fh_v)
    logger.addHandler(fh_q)
