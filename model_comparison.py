from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy
import logging
import os

from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train import cross_validate


DATASETS = OrderedDict()
DATASETS['freesolv'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/freesolv.csv', 10, 'rmse')
DATASETS['delaney'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/delaney.csv', 10, 'rmse')
DATASETS['lipo'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/lipo.csv', 10, 'rmse')
DATASETS['pdbbind_full'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/pdbbind_full.csv', 10, 'rmse')
DATASETS['pdbbind_core'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/pdbbind_core.csv', 10, 'rmse')
DATASETS['pdbbind_refined'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/pdbbind_refined.csv', 10, 'rmse')
DATASETS['qm7'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/qm7.csv', 10, 'mae')
DATASETS['qm8'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/qm8.csv', 10, 'mae')
DATASETS['qm9'] = ('regression', '/data/rsg/chemistry/yangk/chemprop/data/qm9.csv', 3, 'mae')
DATASETS['pcba'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/pcba.csv', 3, 'prc-auc')
DATASETS['muv'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/muv.csv', 3, 'prc-auc')
DATASETS['hiv'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/HIV.csv', 3, 'auc')
DATASETS['bace'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/bace.csv', 10, 'auc')
DATASETS['bbbp'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/BBBP.csv', 10, 'auc')
DATASETS['tox21'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/tox21.csv', 10, 'auc')
DATASETS['toxcast'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/toxcast.csv', 10, 'auc')
DATASETS['sider'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/sider.csv', 10, 'auc')
DATASETS['clintox'] = ('classification', '/data/rsg/chemistry/yangk/chemprop/data/clintox.csv', 10, 'auc')
DATASETS['chembl'] = ('classification', '/data/rsg/chemistry/yangk/chembl/chembl_full.csv', 3, 'auc')

RDKIT_NORMALIZED_FEATURES_DIR = '/data/rsg/chemistry/yangk/saved_features'


def create_train_logger() -> logging.Logger:
    train_logger = logging.getLogger('train')
    train_logger.setLevel(logging.DEBUG)
    train_logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    train_logger.addHandler(ch)

    return train_logger


TRAIN_LOGGER = create_train_logger()


# TODO: change to write results as a CSV for easier processing
def run_comparison(experiment_args: Namespace,
                   logger: logging.Logger,
                   features_dir: str = None):
    for dataset_name in experiment_args.datasets:
        dataset_type, dataset_path, num_folds, metric = DATASETS[dataset_name]
        logger.info(dataset_name)

        # Set up args
        args = deepcopy(experiment_args)
        args.data_path = dataset_path
        args.dataset_type = dataset_type
        args.save_dir = os.path.join(args.save_dir, dataset_name)
        args.num_folds = num_folds
        args.metric = metric
        if features_dir is not None:
            args.features_path = [os.path.join(features_dir, dataset_name + '.pckl')]
        modify_train_args(args)

        # Set up logging for training
        os.makedirs(args.save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
        fh.setLevel(logging.DEBUG)

        # Cross validate
        TRAIN_LOGGER.addHandler(fh)
        mean_score, std_score = cross_validate(args, TRAIN_LOGGER)
        TRAIN_LOGGER.removeHandler(fh)

        # Record results
        logger.info(f'{mean_score} +/- {std_score} {metric}')
        temp_model = build_model(args)
        logger.info(f'num params: {param_count(temp_model):,}')


def create_logger(name: str, save_path: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    save_dir = os.path.dirname(save_path)
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    fh = logging.FileHandler(save_path)
    fh.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(fh)
    
    return logger


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--log_name', type=str, default='gs.log',
                        help='Name of file where model comparison results will be saved')
    parser.add_argument('--experiments', type=str, nargs='*', default=['all'],
                        help='Which experiments to run')
    parser.add_argument('--datasets', type=str, nargs='+', default=list(DATASETS.keys()), choices=list(DATASETS.keys()),
                        help='Which datasets to perform a grid search on')
    args = parser.parse_args()

    log_path = os.path.join(args.save_dir, args.log_name)

    logger = create_logger(name='model_comparison', save_path=log_path)

    if 'all' in args.experiments or 'base' in args.experiments:
        logger.info('base')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'base')
        run_comparison(experiment_args, logger)

    # if 'all' in args.experiments or 'virtual_edges' in args.experiments:
    #     logger.info('virtual edges')
    #     experiment_args = deepcopy(args)
    #     experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'virtual_edges')
    #     experiment_args.virtual_edges = True
    #     run_comparison(experiment_args, logger)
    #
    # if 'all' in args.experiments or 'master_node' in args.experiments:
    #     logger.info('master node')
    #     experiment_args = deepcopy(args)
    #     experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'master_node')
    #     experiment_args.master_node = True
    #     experiment_args.master_dim = experiment_args.hidden_size
    #     experiment_args.use_master_as_output = True
    #     run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'deepset' in args.experiments:
        logger.info('deepset')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'deepset')
        experiment_args.deepset = True
        experiment_args.ffn_num_layers = 1
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'attention' in args.experiments:
        logger.info('attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'attention')
        experiment_args.attention = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'message_attention' in args.experiments:
        logger.info('message attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'message_attention')
        experiment_args.message_attention = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'global_attention' in args.experiments:
        logger.info('global attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'global_attention')
        experiment_args.global_attention = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'diff_depth_weights' in args.experiments:
        logger.info('diff depth weights')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'diff_depth_weights')
        experiment_args.diff_depth_weights = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'layers_per_message' in args.experiments:
        logger.info('layers per message')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'layers_per_message')
        experiment_args.layers_per_message = 2
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'layer_norm' in args.experiments:
        logger.info('layer norm')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'layer_norm')
        experiment_args.layer_norm = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'undirected' in args.experiments:
        logger.info('undirected')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'undirected')
        experiment_args.undirected = True
        run_comparison(experiment_args, logger)
    
    if 'all' in args.experiments or 'scheduler_decay' in args.experiments:
        logger.info('scheduler decay')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'scheduler_decay')
        experiment_args.scheduler = 'decay'
        experiment_args.init_lr = [1e-3]
        run_comparison(experiment_args, logger)
    
    if 'all' in args.experiments or 'rdkit_normalized_features' in args.experiments:
        logger.info('rdkit normalized features')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'rdkit_normalized_features')
        experiment_args.no_features_scaling = True
        run_comparison(experiment_args, logger, features_dir=RDKIT_NORMALIZED_FEATURES_DIR)

    if 'all' in args.experiments or 'nonscaled_targets' in args.experiments:
        logger.info('nonscaled targets')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'nonscaled_targets')
        experiment_args.no_target_scaling = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'class_balance' in args.experiments:
        logger.info('class_balance')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'class_balance')
        experiment_args.class_balance = True
        run_comparison(experiment_args, logger)

    if 'all' in args.experiments or 'atom_messages' in args.experiments:
        logger.info('atom messages')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'atom_messages')
        experiment_args.atom_messages = True
        from chemprop.features.featurization import clear_cache  # needed b/c cache is different for atom messages
        clear_cache()
        run_comparison(experiment_args, logger)
        clear_cache()

    # python model_comparison.py --save_dir logging_dir --log_name gs.log --experiments base --datasets delaney --quiet
