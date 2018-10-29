from argparse import ArgumentParser, Namespace
import os
from tempfile import TemporaryDirectory

import torch


def add_predict_args(parser: ArgumentParser):
    """Add predict arguments to an ArgumentParser."""
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--preds_path', type=str, required=True,
                        help='Path to CSV file where predictions will be saved')


def add_hyper_opt_args(parser: ArgumentParser):
    """Add hyperparameter optimization arguments to an ArgumentParser."""
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory where results will be saved')
    parser.add_argument('--port', type=int, default=9090,
                        help='Port for HpBandSter to use')
    parser.add_argument('--min_budget', type=int, default=5,
                        help='Minimum budget (number of iterations during training) to use')
    parser.add_argument('--max_budget', type=int, default=45,
                        help='Maximum budget (number of iterations during training) to use')
    parser.add_argument('--eta', type=int, default=2,
                        help='Factor by which to cut number of trials (1/eta trials remain)')
    parser.add_argument('--n_iterations', type=int, default=16,
                        help='Number of iterations of BOHB algorithm')


def add_train_args(parser: ArgumentParser):
    """Add training arguments to an ArgumentParser."""
    # General arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--vocab_path', type=str,
                        help='Path to .vocab file if using jtnn')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, choices=['morgan'],
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str,
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--predict_features', action='store_true', default=False,
                        help='Pre-train by predicting the additional features rather than the task values')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--load_encoder_only', action='store_true', default=False,
                        help='If a checkpoint_dir is specified for training, only loads weights from encoder'
                             'and not from the final feed-forward network')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['classification', 'regression', 'regression_with_binning'],
                        help='Type of dataset, i.e. classification (cls) or regression (reg).'
                             'This determines the loss function used during training.')
    parser.add_argument('--num_bins', type=int, default=20,
                        help='Number of bins for regression with binning')
    parser.add_argument('--num_chunks', type=int, default=1,
                        help='Specify > 1 if your dataset is really big')
    parser.add_argument('--chunk_temp_dir', type=str, default='temp_chunks',
                        help='temp dir to store chunks in')
    parser.add_argument('--memoize_chunks', action='store_true', default=False,
                        help='store memo dicts for mol2graph in chunk_temp_dir when chunking, at large disk space cost')
    parser.add_argument('--separate_test_set', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--scaffold_split', action='store_true', default=False,
                        help='Whether to split train/val/test by molecular scaffold instead of randomly')
    parser.add_argument('--scaffold_split_one', action='store_true', default=False,
                        help='Whether to split train/val/test such that each has all molecules from'
                             'exactly one scaffold (largest in train, second largest in val, third largest in test')
    parser.add_argument('--metric', type=str, default=None, choices=['auc', 'prc-auc', 'rmse', 'mae', 'r2', 'accuracy'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, 
                        help='Optional file of fold labels')
    parser.add_argument('--test_fold_index', type=int,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--log_frequency', type=int, default=10,
                        help='The number of batches between each logging of the training loss')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--labels_to_show', type=str, nargs='+',
                        help='List of targets to show individual scores for, if specified')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--truncate_outliers', action='store_true', default=False,
                        help='Truncates outliers in the training set to improve training stability'
                             '(All values outside mean ± 3 * std are truncated to equal mean ± 3 * std)')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--lr_scaler', type=float, default=1.0,
                        help='Amount by which to scale init_lr, max_lr, and final_lr (for convenience)')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='Maximum gradient norm when performing gradient clipping')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='L2 penalty on optimizer to keep parameter norms small')

    # Model arguments
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--layer_norm', action='store_true', default=False,
                        help='Add layer norm after each message passing step')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU', choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh'],
                        help='Activation function')
    parser.add_argument('--attention', action='store_true', default=False,
                        help='Perform self attention over the atoms in a molecule')
    parser.add_argument('--message_attention', action='store_true', default=False,
                        help='Perform attention over messages.')
    parser.add_argument('--global_attention', action='store_true', default=False,
                        help='True to perform global attention across all messages on each message passing step')
    parser.add_argument('--message_attention_heads', type=int, default=1,
                        help='Number of heads to use for message attention')
    parser.add_argument('--master_node', action='store_true', default=False,
                        help='Add a master node to exchange information more easily')
    parser.add_argument('--master_dim', type=int, default=600,
                        help='Number of dimensions for master node state')
    parser.add_argument('--use_master_as_output', action='store_true', default=False,
                        help='Use master node state as output')
    parser.add_argument('--addHs', action='store_true', default=False,
                        help='Explicitly adds hydrogens to the molecular graph')
    parser.add_argument('--three_d', action='store_true', default=False,
                        help='Adds 3D coordinates to atom and bond features')
    parser.add_argument('--virtual_edges', action='store_true', default=False,
                        help='Adds virtual edges between non-bonded atoms')
    parser.add_argument('--drop_virtual_edges', action='store_true', default=False,
                        help='Randomly drops O(n_atoms) virtual edges so O(n_atoms) edges total instead of O(n_atoms^2)')
    parser.add_argument('--learn_virtual_edges', action='store_true', default=False,
                        help='Learn which virtual edges to add, limited to O(n_atoms)')
    parser.add_argument('--deepset', action='store_true', default=False,
                        help='Modify readout function to perform a Deep Sets set operation using linear layers')
    parser.add_argument('--set2set', action='store_true', default=False,
                        help='Modify readout function to perform a set2set operation using an RNN')
    parser.add_argument('--set2set_iters', type=int, default=3,
                        help='Number of set2set RNN iterations to perform')
    parser.add_argument('--jtnn', action='store_true', default=False,
                        help='Build junction tree and perform message passing over both original graph and tree')
    parser.add_argument('--more_ffn_capacity', action='store_true', default=False,
                        help='Give more capacity to the output layers after the graph network')
    parser.add_argument('--ffn_input_dropout', type=float, default=0.2,
                        help='Input dropout for higher-capacity FFN')
    parser.add_argument('--ffn_dropout', type=float, default=0.5,
                        help='Dropout for higher-capacity FFN')
    parser.add_argument('--ffn_hidden_dim', type=int, default=600,
                        help='Hidden dim for higher-capacity FFN')
    parser.add_argument('--adversarial', action='store_true', default=False,
                        help='Adversarial scaffold regularization')
    parser.add_argument('--wgan_beta', type=float, default=10,
                        help='Multiplier for WGAN gradient penalty')
    parser.add_argument('--gan_d_per_g', type=int, default=5,
                        help='GAN discriminator training iterations per generator training iteration')
    parser.add_argument('--gan_lr_mult', type=float, default=0.1,
                        help='Multiplier for GAN generator learning rate')
    parser.add_argument('--gan_use_scheduler', action='store_true', default=False,
                        help='Use noam scheduler for GAN optimizers')
    parser.add_argument('--moe', action='store_true', default=False,
                        help='Use mixture of experts model')
    parser.add_argument('--lambda_moe', type=float, default=0.1,
                        help='Multiplier for moe vs mtl loss')
    parser.add_argument('--lambda_critic', type=float, default=1.0,
                        help='Multiplier for critic loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.001,
                        help='Multiplier for entropy regularization')
    parser.add_argument('--m_rank', type=int, default=100,
                        help='Mahalanobis matrix rank in moe model')
    parser.add_argument('--num_sources', type=int, default=10,
                        help='Number of source tasks for moe')


def modify_hyper_opt_args(args: Namespace):
    """Modifies and validates hyperparameter optimization arguments."""
    os.makedirs(args.results_dir, exist_ok=True)


def update_args_from_checkpoint_dir(args: Namespace):
    """Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size."""
    if args.checkpoint_dir is None:
        args.checkpoint_paths = None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname == 'model.pt':
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)


def modify_train_args(args: Namespace):
    """Modifies and validates training arguments."""
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda

    if args.metric is None:
        args.metric = 'auc' if args.dataset_type == 'classification' else 'rmse'

    if not (args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy'] or
            (args.dataset_type == 'regression' or args.dataset_type == 'regression_with_binning') and args.metric in ['rmse', 'mae', 'r2']):
        raise ValueError('Metric "{}" invalid for dataset type "{}".'.format(args.metric, args.dataset_type))

    args.minimize_score = args.metric in ['rmse', 'mae']

    update_args_from_checkpoint_dir(args)

    if args.jtnn:
        if not hasattr(args, 'vocab_path'):
            raise ValueError('Must provide vocab_path when using jtnn')
        elif not os.path.exists(args.vocab_path):
            raise ValueError('Vocab path "{}" does not exist'.format(args.vocab_path))

    args.use_input_features = args.features_generator or args.features_path

    if args.predict_features:
        assert args.features_generator or args.features_path
        args.use_input_features = False

    args.init_lr *= args.lr_scaler
    args.max_lr *= args.lr_scaler
    args.final_lr *= args.lr_scaler
    del args.lr_scaler


def parse_hyper_opt_args() -> Namespace:
    """Parses arguments for hyperparameter optimization (includes training arguments)."""
    parser = ArgumentParser()
    add_train_args(parser)
    add_hyper_opt_args(parser)
    args = parser.parse_args()
    modify_train_args(args)
    modify_hyper_opt_args(args)

    return args


def parse_train_args() -> Namespace:
    """Parses arguments for training (includes modifying/validating arguments)."""
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args
