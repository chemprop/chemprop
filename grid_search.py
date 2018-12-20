from argparse import ArgumentParser, Namespace
from copy import deepcopy
import logging
import os
from typing import Tuple

import numpy as np

from chemprop.models import build_model
from chemprop.nn_utils import param_count
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.train import cross_validate
from chemprop.utils import set_logger


DATASETS = [('freesolv', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/freesolv.csv', 10, 'rmse'), 
            ('delaney', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/delaney.csv', 10, 'rmse'),
            ('lipo', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/lipo.csv', 10, 'rmse'),
            ('qm7', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/qm7.csv', 10, 'mae'),
            ('qm8', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/qm8.csv', 10, 'mae'),
            ('qm9', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/qm9.csv', 3, 'mae'),
            ('PCBA', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/pcba.csv', 3, 'prc-auc'),
            ('MUV', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/muv.csv', 3, 'prc-auc'),
            ('HIV', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/HIV.csv', 3, 'auc'),
            ('PDBbind', 'regression', '/data/rsg/chemistry/yangk/chemprop/data/pdbbind.csv', 10, 'rmse'),
            ('BACE', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/bace.csv', 10, 'auc'),
            ('BBBP', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/BBBP.csv', 10, 'auc'),
            ('tox21', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/tox21.csv', 10, 'auc'),
            ('toxcast', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/toxcast.csv', 10, 'auc'),
            ('SIDER', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/sider.csv', 10, 'auc'),
            ('clintox', 'classification', '/data/rsg/chemistry/yangk/chemprop/data/clintox.csv', 10, 'auc'),
            ('ChEMBL', 'classification', '/data/rsg/chemistry/yangk/chembl/chembl_full.csv', 3, 'auc')]

gslogger = logging.getLogger('grid_search')
gslogger.setLevel(logging.DEBUG)

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def run_all_datasets(experiment_args: Namespace, gslogger: logging.Logger):
    for dataset_name, dataset_type, dataset_path, num_folds, metric in DATASETS:
        gslogger.info(dataset_name)

        # Set up args
        args = deepcopy(experiment_args)
        args.data_path = dataset_path
        args.dataset_type = dataset_type
        args.save_dir = os.path.join(args.save_dir, dataset_name)
        args.num_folds = num_folds
        args.metric = metric
        modify_train_args(args)

        os.makedirs(args.save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
        fh.setLevel(logging.DEBUG)

        logger.addHandler(fh)
        mean_score, std_score = cross_validate(args, logger)
        logger.removeHandler(fh)

        gslogger.info('{} +/- {} {}'.format(mean_score, std_score, metric))
        temp_model = build_model(args)
        gslogger.info('num params: {:,}'.format(param_count(temp_model)))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--log_name', type=str, default='gs.log',
                        help='Name of file where grid search results will be saved')
    parser.add_argument('--experiments', type=str, nargs='*', default=['all'],
                        help='Which experiments to run')
    args = parser.parse_args()

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    os.makedirs(args.save_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    fh.setLevel(logging.DEBUG)

    gslogger.addHandler(ch)
    gslogger.addHandler(fh)

    # TODO add atom experiment

    if 'all' in args.experiments or 'base' in args.experiments:
        gslogger.info('base')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'base')
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'virtual_edges' in args.experiments:
        gslogger.info('virtual edges')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'virtual_edges')
        experiment_args.virtual_edges = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'master_node' in args.experiments:
        gslogger.info('master node')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'master_node')
        experiment_args.master_node = True
        experiment_args.master_dim = experiment_args.hidden_size
        experiment_args.use_master_as_output = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'deepset' in args.experiments:
        gslogger.info('deepset')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'deepset')
        experiment_args.deepset = True
        experiment_args.ffn_num_layers = 1
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'attention' in args.experiments:
        gslogger.info('attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'attention')
        experiment_args.attention = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'message_attention' in args.experiments:
        gslogger.info('message attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'message_attention')
        experiment_args.message_attention = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'global_attention' in args.experiments:
        gslogger.info('global attention')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'global_attention')
        experiment_args.global_attention = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'diff_depth_weights' in args.experiments:
        gslogger.info('diff depth weights')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'diff_depth_weights')
        experiment_args.diff_depth_weights = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'layers_per_message' in args.experiments:
        gslogger.info('layers per message')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'layers_per_message')
        experiment_args.layers_per_message = 2
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'layer_norm' in args.experiments:
        gslogger.info('layer norm')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'layer_norm')
        experiment_args.layer_norm = True
        run_all_datasets(experiment_args, gslogger)

    if 'all' in args.experiments or 'undirected' in args.experiments:
        gslogger.info('undirected')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'undirected')
        experiment_args.undirected = True
        run_all_datasets(experiment_args, gslogger)
    
    if 'all' in args.experiments or 'scheduler_decay' in args.experiments:
        gslogger.info('scheduler_decay')
        experiment_args = deepcopy(args)
        experiment_args.save_dir = os.path.join(experiment_args.save_dir, 'scheduler_decay')
        experiment_args.scheduler = 'decay'
        experiment_args.init_lr = [1e-3]
        run_all_datasets(experiment_args, gslogger)

    # python grid_search.py --data_path anything --dataset_type anything --save_dir logging dir --quiet
