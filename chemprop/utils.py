from collections import Counter
import logging
import math
import os
from typing import Callable, List, Tuple
from argparse import Namespace

import numpy as np
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer, SGD
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR

from chemprop.data import StandardScaler
from chemprop.models import build_model
from chemprop.nn_utils import MockLR, NoamLR


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
        'data_scaler': {
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
                    logger: logging.Logger = None) -> nn.Module:
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded model.
    """
    debug = logger.debug if logger is not None else print

    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    load_encoder_only = current_args.load_encoder_only if current_args is not None else False

    # Build model
    model = build_model(args)
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        if load_encoder_only and 'encoder' not in param_name:
            continue

        if param_name not in model_state_dict:
            debug('Pretrained parameter "{}" cannot be found in model parameters.'.format(param_name))
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug('Pretrained parameter "{}" of shape {} does not match corresponding '
                  'model parameter of shape {}.'.format(param_name,
                                                        loaded_state_dict[param_name].shape,
                                                        model_state_dict[param_name].shape))
        else:
            debug('Loading pretrained parameter "{}".'.format(param_name))
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
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data scaler and the features scaler.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    data_scaler = StandardScaler(state['data_scaler']['means'],
                                 state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return data_scaler, features_scaler


def load_args(path: str) -> Namespace:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The arguments Namespace that the model was trained with.
    """
    return torch.load(path, map_location=lambda storage, loc: storage)['args']


def get_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression" or "regression_with_binning").
    :return: A PyTorch loss function.
    """
    if args.dataset_type == 'classification':
        return nn.BCELoss(reduction='none')
    
    if args.dataset_type == 'regression_with_binning':
        return nn.CrossEntropyLoss(reduction='none')

    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')
    
    if args.dataset_type == 'unsupervised':
        return nn.CrossEntropyLoss(reduction='none')

    if args.dataset_type == 'bert_pretraining':
        if args.bert_vocab_func == 'feature_vector':
            # TODO a lot of the targets are actually classification targets, though...?
            return nn.MSELoss(reduction='none')
        else:
            return nn.CrossEntropyLoss(reduction='none')

    raise ValueError('Dataset type "{}" not supported.'.format(args.dataset_type))


def prc_auc(targets: List[int], preds: List[float]) -> float:
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float], preds: List[float]) -> float:
    return math.sqrt(mean_squared_error(targets, preds))


def accuracy(targets: List[int], preds: List[float]) -> float:
    hard_preds = [1 if p > 0.5 else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def argmax_accuracy(targets: List[int], preds: List[List[float]]) -> float:
    hard_preds = np.argmax(preds, axis=1)
    return accuracy_score(targets, hard_preds)


def majority_baseline_accuracy(targets: List[int], *args) -> float:
    counter = Counter(targets)
    return counter.most_common()[0][1] / len(targets)


def get_metric_func(args: Namespace) -> Callable:
    """
    Gets the metric function corresponding to a given metric name.

    :param args: Namespace containing args.metric
    :return: A metric function which takes as arguments a list of targets and a list of predictions.
    """
    metric = args.metric

    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score
    
    if metric == 'accuracy':
        return accuracy

    if metric == 'argmax_accuracy':
        return argmax_accuracy
    
    if metric == 'log_loss':
        # only supported for unsupervised and bert_pretraining
        num_labels = args.unsupervised_n_clusters if args.dataset_type == 'unsupervised' else args.vocab.output_size

        def metric_func(targets: List[int], preds: List[List[float]]) -> float:
            return log_loss(targets, preds, labels=range(num_labels))

        return metric_func

    if metric == 'majority_baseline_accuracy':
        return majority_baseline_accuracy

    raise ValueError('Metric "{}" not supported.'.format(metric))


def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    if args.separate_ffn_lr:
        params = [
            {'params': model.encoder.parameters(), 'lr': args.init_lr[0], 'weight_decay': args.weight_decay[0]},
            {'params': model.ffn.parameters(), 'lr': args.init_lr[1], 'weight_decay': args.weight_decay[1]}
        ]
    else:
        params = [{'params': model.parameters(), 'lr': args.init_lr[0], 'weight_decay': args.weight_decay[0]}]

    if args.optimizer == 'Adam':
        return Adam(params)

    if args.optimizer == 'SGD':
        return SGD(params)

    raise ValueError('Optimizer "{}" not supported.'.format(args.optimizer))


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, total_epochs: List[int] = None) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    if args.scheduler == 'noam':
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=total_epochs or [args.epochs] * args.num_lrs,
            steps_per_epoch=args.train_data_size // args.batch_size,
            init_lr=args.init_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )

    if args.scheduler == 'none':
        return MockLR(optimizer=optimizer, lr=args.init_lr)

    if args.scheduler == 'decay':
        return ExponentialLR(optimizer, args.lr_decay_rate)

    raise ValueError('Learning rate scheduler "{}" not supported.'.format(args.scheduler))


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
