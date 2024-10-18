from copy import deepcopy
from io import StringIO
import json
import logging
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from configargparse import ArgumentError, ArgumentParser, Namespace
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Column, Table
import torch
import torch.nn as nn

from chemprop.cli.common import (
    add_common_args,
    find_models,
    process_common_args,
    validate_common_args,
)
from chemprop.cli.conf import NOW
from chemprop.cli.utils import (
    LookupAction,
    Subcommand,
    build_data_from_files,
    get_column_names,
    make_dataset,
    parse_indices,
)
from chemprop.cli.utils.args import uppercase
from chemprop.data import (
    MoleculeDataset,
    MolGraphDataset,
    MulticomponentDataset,
    ReactionDatapoint,
    SplitType,
    build_dataloader,
    make_split_indices,
    split_data_by_indices,
)
from chemprop.data.datasets import _MolGraphDatasetMixin
from chemprop.models import MPNN, MulticomponentMPNN, save_model
from chemprop.nn import AggregationRegistry, LossFunctionRegistry, MetricRegistry, PredictorRegistry
from chemprop.nn.message_passing import (
    AtomMessagePassing,
    BondMessagePassing,
    MulticomponentMessagePassing,
)
from chemprop.nn.transforms import GraphTransform, ScaleTransform, UnscaleTransform
from chemprop.nn.utils import Activation
from chemprop.utils import Factory

logger = logging.getLogger(__name__)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "Train a chemprop model."
    parser = None

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        parser = add_train_args(parser)
        cls.parser = parser
        return parser

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        validate_common_args(args)
        args = process_train_args(args)
        validate_train_args(args)

        args.output_dir.mkdir(exist_ok=True, parents=True)
        config_path = args.output_dir / "config.toml"
        save_config(cls.parser, args, config_path)
        main(args)


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--config-path",
        type=Path,
        is_config_file=True,
        help="Path to a configuration file (command line arguments override values in the configuration file)",
    )
    parser.add_argument(
        "-i",
        "--data-path",
        type=Path,
        help="Path to an input CSV file containing SMILES and the associated target values",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--save-dir",
        type=Path,
        help="Directory where training outputs will be saved (defaults to ``CURRENT_DIRECTORY/chemprop_training/STEM_OF_INPUT/TIME_STAMP``)",
    )
    parser.add_argument(
        "--remove-checkpoints",
        action="store_true",
        help="Remove intermediate checkpoint files after training is complete.",
    )

    # TODO: Add in v2.1; see if we can tell lightning how often to log training loss
    # parser.add_argument(
    #     "--log-frequency",
    #     type=int,
    #     default=10,
    #     help="The number of batches between each logging of the training loss.",
    # )

    transfer_args = parser.add_argument_group("transfer learning args")
    transfer_args.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        help="Path to checkpoint(s) or model file(s) for loading and overwriting weights. Accepts a single pre-trained model checkpoint (.ckpt), a single model file (.pt), a directory containing such files, or a list of paths and directories. If a directory is provided, it will recursively search for and use all (.pt) files found for prediction.",
    )
    transfer_args.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze the message passing layer from the checkpoint model (specified by ``--checkpoint``).",
    )
    transfer_args.add_argument(
        "--model-frzn",
        help="Path to model checkpoint file to be loaded for overwriting and freezing weights. By default, all MPNN weights are frozen with this option.",
    )
    transfer_args.add_argument(
        "--frzn-ffn-layers",
        type=int,
        default=0,
        help="Freeze the first ``n`` layers of the FFN from the checkpoint model (specified by ``--checkpoint``). The message passing layer should also be frozen with ``--freeze-encoder``.",
    )
    # transfer_args.add_argument(
    #     "--freeze-first-only",
    #     action="store_true",
    #     help="Determines whether or not to use checkpoint_frzn for just the first encoder. Default (False) is to use the checkpoint to freeze all encoders. (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)",
    # )

    # TODO: Add in v2.1
    # parser.add_argument(
    #     "--resume-experiment",
    #     action="store_true",
    #     help="Whether to resume the experiment. Loads test results from any folds that have already been completed and skips training those folds.",
    # )
    # parser.add_argument(
    #     "--config-path",
    #     help="Path to a :code:`.json` file containing arguments. Any arguments present in the config file will override arguments specified via the command line or by the defaults.",
    # )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of models in ensemble for each splitting of data",
    )

    # TODO: Add in v2.2
    # abt_args = parser.add_argument_group("atom/bond target args")
    # abt_args.add_argument(
    #     "--is-atom-bond-targets",
    #     action="store_true",
    #     help="Whether this is atomic/bond properties prediction.",
    # )
    # abt_args.add_argument(
    #     "--no-adding-bond-types",
    #     action="store_true",
    #     help="Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used with the :code:`is_atom_bond_targets`.",
    # )
    # abt_args.add_argument(
    #     "--keeping-atom-map",
    #     action="store_true",
    #     help="Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with the :code:`is_atom_bond_targets`.",
    # )
    # abt_args.add_argument(
    #     "--no-shared-atom-bond-ffn",
    #     action="store_true",
    #     help="Whether the FFN weights for atom and bond targets should be independent between tasks.",
    # )
    # abt_args.add_argument(
    #     "--weights-ffn-num-layers",
    #     type=int,
    #     default=2,
    #     help="Number of layers in FFN for determining weights used in constrained targets.",
    # )

    mp_args = parser.add_argument_group("message passing")
    mp_args.add_argument(
        "--message-hidden-dim", type=int, default=300, help="Hidden dimension of the messages"
    )
    mp_args.add_argument(
        "--message-bias", action="store_true", help="Add bias to the message passing layers"
    )
    mp_args.add_argument("--depth", type=int, default=3, help="Number of message passing steps")
    mp_args.add_argument(
        "--undirected",
        action="store_true",
        help="Pass messages on undirected bonds/edges (always sum the two relevant bond vectors)",
    )
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--mpn-shared",
        action="store_true",
        help="Whether to use the same message passing neural network for all input molecules (only relevant if ``number_of_molecules`` > 1)",
    )
    mp_args.add_argument(
        "--activation",
        type=uppercase,
        default="RELU",
        choices=list(Activation.keys()),
        help="Activation function in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--aggregation",
        "--agg",
        default="norm",
        action=LookupAction(AggregationRegistry),
        help="Aggregation mode to use during graph predictor",
    )
    mp_args.add_argument(
        "--aggregation-norm",
        type=float,
        default=100,
        help="Normalization factor by which to divide summed up atomic features for ``norm`` aggregation",
    )
    mp_args.add_argument(
        "--atom-messages", action="store_true", help="Pass messages on atoms rather than bonds."
    )

    # TODO: Add in v2.1
    # mpsolv_args = parser.add_argument_group("message passing with solvent")
    # mpsolv_args.add_argument(
    #     "--reaction-solvent",
    #     action="store_true",
    #     help="Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.",
    # )
    # mpsolv_args.add_argument(
    #     "--bias-solvent",
    #     action="store_true",
    #     help="Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True.",
    # )
    # mpsolv_args.add_argument(
    #     "--hidden-size-solvent",
    #     type=int,
    #     default=300,
    #     help="Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True.",
    # )
    # mpsolv_args.add_argument(
    #     "--depth-solvent",
    #     type=int,
    #     default=3,
    #     help="Number of message passing steps for solvent if :code:`reaction_solvent` is True.",
    # )

    ffn_args = parser.add_argument_group("FFN args")
    ffn_args.add_argument(
        "--ffn-hidden-dim", type=int, default=300, help="Hidden dimension in the FFN top model"
    )
    ffn_args.add_argument(  # TODO: the default in v1 was 2. (see weights_ffn_num_layers option) Do we really want the default to now be 1?
        "--ffn-num-layers", type=int, default=1, help="Number of layers in FFN top model"
    )
    # TODO: Decide if we want to implment this in v2
    # ffn_args.add_argument(
    #     "--features-only",
    #     action="store_true",
    #     help="Use only the additional features in an FFN, no graph network.",
    # )

    extra_mpnn_args = parser.add_argument_group("extra MPNN args")
    extra_mpnn_args.add_argument(
        "--no-batch-norm",
        action="store_true",
        help="Turn off batch normalization after aggregation",
    )
    extra_mpnn_args.add_argument(
        "--multiclass-num-classes",
        type=int,
        default=3,
        help="Number of classes when running multiclass classification",
    )
    # TODO: Add in v2.1
    # extra_mpnn_args.add_argument(
    #     "--spectral-activation",
    #     default="exp",
    #     choices=["softplus", "exp"],
    #     help="Indicates which function to use in task_type spectra training to constrain outputs to be positive.",
    # )

    train_data_args = parser.add_argument_group("training input data args")
    train_data_args.add_argument(
        "-w",
        "--weight-column",
        help="Name of the column in the input CSV containing individual data weights",
    )
    train_data_args.add_argument(
        "--target-columns",
        nargs="+",
        help="Name of the columns containing target values (by default, uses all columns except the SMILES column and the ``ignore_columns``)",
    )
    train_data_args.add_argument(
        "--ignore-columns",
        nargs="+",
        help="Name of the columns to ignore when ``target_columns`` is not provided",
    )
    train_data_args.add_argument(
        "--no-cache",
        action="store_true",
        help="Turn off caching the featurized ``MolGraph`` s at the beginning of training",
    )
    # TODO: Add in v2.1
    # train_data_args.add_argument(
    #     "--spectra-phase-mask-path",
    #     help="Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions.",
    # )

    train_args = parser.add_argument_group("training args")
    train_args.add_argument(
        "-t",
        "--task-type",
        default="regression",
        action=LookupAction(PredictorRegistry),
        help="Type of dataset (determines the default loss function used during training, defaults to ``regression``)",
    )
    train_args.add_argument(
        "-l",
        "--loss-function",
        action=LookupAction(LossFunctionRegistry),
        help="Loss function to use during training (will use the default loss function for the given task type if not specified)",
    )
    train_args.add_argument(
        "--v-kl",
        "--evidential-regularization",
        type=float,
        default=0.0,
        help="Specify the value used in regularization for evidential loss function. The default value recommended by Soleimany et al. (2021) is 0.2. However, the optimal value is dataset-dependent, so it is recommended that users test different values to find the best value for their model.",
    )

    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="Evidential regularization epsilon"
    )
    # TODO: Add in v2.1
    # train_args.add_argument(  # TODO: Is threshold the same thing as the spectra target floor? I'm not sure but combined them.
    #     "-T",
    #     "--threshold",
    #     "--spectra-target-floor",
    #     type=float,
    #     default=1e-8,
    #     help="spectral threshold limit. v1 help string: Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values.",
    # )
    train_args.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        action=LookupAction(MetricRegistry),
        help="Specify the evaluation metrics. If unspecified, chemprop will use the following metrics for given dataset types: regression -> ``rmse``, classification -> ``roc``, multiclass -> ``ce`` ('cross entropy'), spectral -> ``sid``. If multiple metrics are provided, the 0-th one will be used for early stopping and checkpointing.",
    )
    train_args.add_argument(
        "--show-individual-scores",
        action="store_true",
        help="Show all scores for individual targets, not just average, at the end.",
    )
    train_args.add_argument(
        "--task-weights",
        nargs="+",
        type=float,
        help="Weights to apply for whole tasks in the loss function",
    )
    train_args.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of epochs during which learning rate increases linearly from ``init_lr`` to ``max_lr`` (afterwards, learning rate decreases exponentially from ``max_lr`` to ``final_lr``)",
    )

    train_args.add_argument("--init-lr", type=float, default=1e-4, help="Initial learning rate.")
    train_args.add_argument("--max-lr", type=float, default=1e-3, help="Maximum learning rate.")
    train_args.add_argument("--final-lr", type=float, default=1e-4, help="Final learning rate.")
    train_args.add_argument("--epochs", type=int, default=50, help="Number of epochs to train over")
    train_args.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of epochs to wait for improvement before early stopping",
    )
    train_args.add_argument(
        "--grad-clip",
        type=float,
        help="Passed directly to the lightning trainer which controls grad clipping (see the ``Trainer()`` docstring for details)",
    )
    train_args.add_argument(
        "--class-balance",
        action="store_true",
        help="Ensures each training batch contains an equal number of positive and negative samples.",
    )

    split_args = parser.add_argument_group("split args")
    split_args.add_argument(
        "--split",
        "--split-type",
        type=uppercase,
        default="RANDOM",
        choices=list(SplitType.keys()),
        help="Method of splitting the data into train/val/test (case insensitive)",
    )
    split_args.add_argument(
        "--split-sizes",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Split proportions for train/validation/test sets",
    )
    split_args.add_argument(
        "--split-key-molecule",
        type=int,
        default=0,
        help="Specify the index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used (e.g., ``scaffold_balanced`` or ``random_with_repeated_smiles``). Note that this index begins with zero for the first molecule.",
    )
    split_args.add_argument(
        "-k",
        "--num-folds",
        type=int,
        default=1,
        help="Number of folds when performing cross validation",
    )
    split_args.add_argument(
        "--save-smiles-splits",
        action="store_true",
        help="Whether to store the SMILES in each train/val/test split",
    )
    split_args.add_argument(
        "--splits-file",
        type=Path,
        help="Path to a JSON file containing pre-defined splits for the input data, formatted as a list of dictionaries with keys ``train``, ``val``, and ``test`` and values as lists of indices or formatted strings (e.g. [0, 1, 2, 4] or '0-2,4')",
    )
    train_data_args.add_argument(
        "--splits-column",
        help="Name of the column in the input CSV file containing ``train``, ``val``, or ``test`` for each row",
    )
    split_args.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Specify the random seed to use when splitting data into train/val/test sets. When ``num_folds`` > 1, the first fold uses this seed and all subsequent folds add 1 to the seed (also used for shuffling data in ``build_dataloader`` when ``shuffle`` is True).",
    )

    parser.add_argument(
        "--pytorch-seed",
        type=int,
        default=None,
        help="Seed for PyTorch randomness (e.g., random initial weights)",
    )

    return parser


def process_train_args(args: Namespace) -> Namespace:
    if args.config_path is None and args.data_path is None:
        raise ArgumentError(argument=None, message="Data path must be provided for training.")

    if args.data_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None, message=f"Input data must be a CSV file. Got {args.data_path}"
        )

    if args.output_dir is None:
        args.output_dir = Path(f"chemprop_training/{args.data_path.stem}/{NOW}")

    if args.epochs != -1 and args.epochs <= args.warmup_epochs:
        raise ArgumentError(
            argument=None,
            message=f"The number of epochs should be higher than the number of epochs during warmup. Got {args.epochs} epochs and {args.warmup_epochs} warmup epochs",
        )

    # TODO: model_frzn is deprecated and then remove in v2.2
    if args.checkpoint is not None and args.model_frzn is not None:
        raise ArgumentError(
            argument=None,
            message="`--checkpoint` and `--model-frzn` cannot be used at the same time.",
        )

    if "--model-frzn" in sys.argv:
        logger.warning(
            "`--model-frzn` is deprecated and will be removed in v2.2. "
            "Please use `--checkpoint` with `--freeze-encoder` instead."
        )

    if args.freeze_encoder and args.checkpoint is None:
        raise ArgumentError(
            argument=None,
            message="`--freeze-encoder` can only be used when `--checkpoint` is used.",
        )

    if args.frzn_ffn_layers > 0:
        if args.checkpoint is None and args.model_frzn is None:
            raise ArgumentError(
                argument=None,
                message="`--frzn-ffn-layers` can only be used when `--checkpoint` or `--model-frzn` (depreciated in v2.1) is used.",
            )
        if args.checkpoint is not None and not args.freeze_encoder:
            raise ArgumentError(
                argument=None,
                message="To freeze the first `n` layers of the FFN via `--frzn-ffn-layers`. The message passing layer should also be frozen with `--freeze-encoder`.",
            )

    if args.class_balance and args.task_type != "classification":
        raise ArgumentError(
            argument=None, message="Class balance is only applicable for classification tasks."
        )

    input_cols, target_cols = get_column_names(
        args.data_path,
        args.smiles_columns,
        args.reaction_columns,
        args.target_columns,
        args.ignore_columns,
        args.splits_column,
        args.weight_column,
        args.no_header_row,
    )

    args.input_columns = input_cols
    args.target_columns = target_cols

    return args


def validate_train_args(args):
    pass


def normalize_inputs(train_dset, val_dset, args):
    multicomponent = isinstance(train_dset, MulticomponentDataset)
    num_components = train_dset.n_components if multicomponent else 1

    X_d_transform = None
    V_f_transforms = [nn.Identity()] * num_components
    E_f_transforms = [nn.Identity()] * num_components
    V_d_transforms = [None] * num_components
    graph_transforms = []

    d_xd = train_dset.d_xd
    d_vf = train_dset.d_vf
    d_ef = train_dset.d_ef
    d_vd = train_dset.d_vd

    if d_xd > 0 and not args.no_descriptor_scaling:
        scaler = train_dset.normalize_inputs("X_d")
        val_dset.normalize_inputs("X_d", scaler)

        scaler = scaler if not isinstance(scaler, list) else scaler[0]

        if scaler is not None:
            logger.info(
                f"Descriptors: loc = {np.array2string(scaler.mean_, precision=3)}, scale = {np.array2string(scaler.scale_, precision=3)}"
            )
            X_d_transform = ScaleTransform.from_standard_scaler(scaler)

    if d_vf > 0 and not args.no_atom_feature_scaling:
        scaler = train_dset.normalize_inputs("V_f")
        val_dset.normalize_inputs("V_f", scaler)

        scalers = [scaler] if not isinstance(scaler, list) else scaler

        for i, scaler in enumerate(scalers):
            if scaler is None:
                continue

            logger.info(
                f"Atom features for mol {i}: loc = {np.array2string(scaler.mean_, precision=3)}, scale = {np.array2string(scaler.scale_, precision=3)}"
            )
            featurizer = (
                train_dset.datasets[i].featurizer if multicomponent else train_dset.featurizer
            )
            V_f_transforms[i] = ScaleTransform.from_standard_scaler(
                scaler, pad=featurizer.atom_fdim - featurizer.extra_atom_fdim
            )

    if d_ef > 0 and not args.no_bond_feature_scaling:
        scaler = train_dset.normalize_inputs("E_f")
        val_dset.normalize_inputs("E_f", scaler)

        scalers = [scaler] if not isinstance(scaler, list) else scaler

        for i, scaler in enumerate(scalers):
            if scaler is None:
                continue

            logger.info(
                f"Bond features for mol {i}: loc = {np.array2string(scaler.mean_, precision=3)}, scale = {np.array2string(scaler.scale_, precision=3)}"
            )
            featurizer = (
                train_dset.datasets[i].featurizer if multicomponent else train_dset.featurizer
            )
            E_f_transforms[i] = ScaleTransform.from_standard_scaler(
                scaler, pad=featurizer.bond_fdim - featurizer.extra_bond_fdim
            )

    for V_f_transform, E_f_transform in zip(V_f_transforms, E_f_transforms):
        graph_transforms.append(GraphTransform(V_f_transform, E_f_transform))

    if d_vd > 0 and not args.no_atom_descriptor_scaling:
        scaler = train_dset.normalize_inputs("V_d")
        val_dset.normalize_inputs("V_d", scaler)

        scalers = [scaler] if not isinstance(scaler, list) else scaler

        for i, scaler in enumerate(scalers):
            if scaler is None:
                continue

            logger.info(
                f"Atom descriptors for mol {i}: loc = {np.array2string(scaler.mean_, precision=3)}, scale = {np.array2string(scaler.scale_, precision=3)}"
            )
            V_d_transforms[i] = ScaleTransform.from_standard_scaler(scaler)

    return X_d_transform, graph_transforms, V_d_transforms


def load_and_use_pretrained_model_scalers(model_path: Path, train_dset, val_dset) -> None:
    if isinstance(train_dset, MulticomponentDataset):
        _model = MulticomponentMPNN.load_from_file(model_path)
        blocks = _model.message_passing.blocks
        train_dsets = train_dset.datasets
        val_dsets = val_dset.datasets
    else:
        _model = MPNN.load_from_file(model_path)
        blocks = [_model.message_passing]
        train_dsets = [train_dset]
        val_dsets = [val_dset]

    for i in range(len(blocks)):
        if isinstance(_model.X_d_transform, ScaleTransform):
            scaler = _model.X_d_transform.to_standard_scaler()
            train_dsets[i].normalize_inputs("X_d", scaler)
            val_dsets[i].normalize_inputs("X_d", scaler)

        if isinstance(blocks[i].graph_transform, GraphTransform):
            if isinstance(blocks[i].graph_transform.V_transform, ScaleTransform):
                V_anti_pad = (
                    train_dsets[i].featurizer.atom_fdim - train_dsets[i].featurizer.extra_atom_fdim
                )
                scaler = blocks[i].graph_transform.V_transform.to_standard_scaler(
                    anti_pad=V_anti_pad
                )
                train_dsets[i].normalize_inputs("V_f", scaler)
                val_dsets[i].normalize_inputs("V_f", scaler)
            if isinstance(blocks[i].graph_transform.E_transform, ScaleTransform):
                E_anti_pad = (
                    train_dsets[i].featurizer.bond_fdim - train_dsets[i].featurizer.extra_bond_fdim
                )
                scaler = blocks[i].graph_transform.E_transform.to_standard_scaler(
                    anti_pad=E_anti_pad
                )
                train_dsets[i].normalize_inputs("E_f", scaler)
                val_dsets[i].normalize_inputs("E_f", scaler)

        if isinstance(blocks[i].V_d_transform, ScaleTransform):
            scaler = blocks[i].V_d_transform.to_standard_scaler()
            train_dsets[i].normalize_inputs("V_d", scaler)
            val_dsets[i].normalize_inputs("V_d", scaler)

    if isinstance(_model.predictor.output_transform, UnscaleTransform):
        scaler = _model.predictor.output_transform.to_standard_scaler()
        train_dset.normalize_targets(scaler)
        val_dset.normalize_targets(scaler)


def save_config(parser: ArgumentParser, args: Namespace, config_path: Path):
    config_args = deepcopy(args)
    for key, value in vars(config_args).items():
        if isinstance(value, Path):
            setattr(config_args, key, str(value))

    for key in ["atom_features_path", "atom_descriptors_path", "bond_features_path"]:
        if getattr(config_args, key) is not None:
            for index, path in getattr(config_args, key).items():
                getattr(config_args, key)[index] = str(path)

    parser.write_config_file(parsed_namespace=config_args, output_file_paths=[str(config_path)])


def save_smiles_splits(args: Namespace, output_dir, train_dset, val_dset, test_dset):
    match (args.smiles_columns, args.reaction_columns):
        case [_, None]:
            column_labels = deepcopy(args.smiles_columns)
        case [None, _]:
            column_labels = deepcopy(args.reaction_columns)
        case _:
            column_labels = deepcopy(args.smiles_columns)
            column_labels.extend(args.reaction_columns)

    train_smis = train_dset.names
    df_train = pd.DataFrame(train_smis, columns=column_labels)
    df_train.to_csv(output_dir / "train_smiles.csv", index=False)

    val_smis = val_dset.names
    df_val = pd.DataFrame(val_smis, columns=column_labels)
    df_val.to_csv(output_dir / "val_smiles.csv", index=False)

    if test_dset is not None:
        test_smis = test_dset.names
        df_test = pd.DataFrame(test_smis, columns=column_labels)
        df_test.to_csv(output_dir / "test_smiles.csv", index=False)


def build_splits(args, format_kwargs, featurization_kwargs):
    """build the train/val/test splits"""
    logger.info(f"Pulling data from file: {args.data_path}")
    all_data = build_data_from_files(
        args.data_path,
        p_descriptors=args.descriptors_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **format_kwargs,
        **featurization_kwargs,
    )

    if args.splits_column is not None:
        df = pd.read_csv(
            args.data_path, header=None if args.no_header_row else "infer", index_col=False
        )
        grouped = df.groupby(df[args.splits_column].str.lower())
        train_indices = grouped.groups.get("train", pd.Index([])).tolist()
        val_indices = grouped.groups.get("val", pd.Index([])).tolist()
        test_indices = grouped.groups.get("test", pd.Index([])).tolist()
        train_indices, val_indices, test_indices = [train_indices], [val_indices], [test_indices]

    elif args.splits_file is not None:
        with open(args.splits_file, "rb") as json_file:
            split_idxss = json.load(json_file)
        train_indices = [parse_indices(d["train"]) for d in split_idxss]
        val_indices = [parse_indices(d["val"]) for d in split_idxss]
        test_indices = [parse_indices(d["test"]) for d in split_idxss]

    else:
        splitting_data = all_data[args.split_key_molecule]
        if isinstance(splitting_data[0], ReactionDatapoint):
            splitting_mols = [datapoint.rct for datapoint in splitting_data]
        else:
            splitting_mols = [datapoint.mol for datapoint in splitting_data]
        train_indices, val_indices, test_indices = make_split_indices(
            splitting_mols, args.split, args.split_sizes, args.data_seed, args.num_folds
        )
        if not (
            SplitType.get(args.split) == SplitType.CV_NO_VAL
            or SplitType.get(args.split) == SplitType.CV
        ):
            train_indices, val_indices, test_indices = (
                [train_indices],
                [val_indices],
                [test_indices],
            )

    train_data, val_data, test_data = split_data_by_indices(
        all_data, train_indices, val_indices, test_indices
    )
    for i_split in range(len(train_data)):
        sizes = [len(train_data[i_split][0]), len(val_data[i_split][0]), len(test_data[i_split][0])]
        logger.info(f"train/val/test split_{i_split} sizes: {sizes}")

    return train_data, val_data, test_data


def summarize(
    target_cols: list[str], task_type: str, dataset: _MolGraphDatasetMixin
) -> tuple[list, list]:
    if task_type in ["regression", "regression-mve", "regression-evidential"]:
        if isinstance(dataset, MulticomponentDataset):
            y = dataset.datasets[0].Y
        else:
            y = dataset.Y
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0)
        y_median = np.nanmedian(y, axis=0)
        mean_dev_abs = np.abs(y - y_mean)
        num_targets = np.sum(~np.isnan(y), axis=0)
        frac_1_sigma = np.sum((mean_dev_abs < y_std), axis=0) / num_targets
        frac_2_sigma = np.sum((mean_dev_abs < 2 * y_std), axis=0) / num_targets

        column_headers = ["Statistic"] + [f"Value ({target_cols[i]})" for i in range(y.shape[1])]
        table_rows = [
            ["Num. smiles"] + [f"{len(y)}" for i in range(y.shape[1])],
            ["Num. targets"] + [f"{num_targets[i]}" for i in range(y.shape[1])],
            ["Num. NaN"] + [f"{len(y) - num_targets[i]}" for i in range(y.shape[1])],
            ["Mean"] + [f"{mean:0.3g}" for mean in y_mean],
            ["Std. dev."] + [f"{std:0.3g}" for std in y_std],
            ["Median"] + [f"{median:0.3g}" for median in y_median],
            ["% within 1 s.d."] + [f"{sigma:0.0%}" for sigma in frac_1_sigma],
            ["% within 2 s.d."] + [f"{sigma:0.0%}" for sigma in frac_2_sigma],
        ]
        return (column_headers, table_rows)
    elif task_type in [
        "classification",
        "classification-dirichlet",
        "multiclass",
        "multiclass-dirichlet",
    ]:
        if isinstance(dataset, MulticomponentDataset):
            y = dataset.datasets[0].Y
        else:
            y = dataset.Y

        mask = np.isnan(y)
        classes = np.sort(np.unique(y[~mask]))

        class_counts = np.stack([(classes[:, None] == y[:, i]).sum(1) for i in range(y.shape[1])])
        class_fracs = class_counts / y.shape[0]
        nan_count = np.nansum(mask, axis=0)
        nan_frac = nan_count / y.shape[0]

        column_headers = ["Class"] + [f"Count/Percent {target_cols[i]}" for i in range(y.shape[1])]

        table_rows = [
            [f"{k}"] + [f"{class_counts[j,i]}/{class_fracs[j,i]:0.0%}" for j in range(y.shape[1])]
            for i, k in enumerate(classes)
        ]

        nan_row = ["NaN"] + [f"{nan_count[i]}/{nan_frac[i]:0.0%}" for i in range(y.shape[1])]
        table_rows.append(nan_row)

        total_row = ["Total"] + [f"{y.shape[0]}/{100.00}%" for i in range(y.shape[1])]
        table_rows.append(total_row)

        return (column_headers, table_rows)
    else:
        raise ValueError(f"unsupported task type! Task type '{args.task_type}' was not recognized.")


def build_table(column_headers: list[str], table_rows: list[str], title: str | None = None) -> str:
    right_justified_columns = [
        Column(header=column_header, justify="right") for column_header in column_headers
    ]
    table = Table(*right_justified_columns, title=title)
    for row in table_rows:
        table.add_row(*row)

    console = Console(record=True, file=StringIO(), width=200)
    console.print(table)
    return console.export_text()


def build_datasets(args, train_data, val_data, test_data):
    """build the train/val/test datasets, where :attr:`test_data` may be None"""
    multicomponent = len(train_data) > 1
    if multicomponent:
        train_dsets = [
            make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
            for data in train_data
        ]
        val_dsets = [
            make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
            for data in val_data
        ]
        train_dset = MulticomponentDataset(train_dsets)
        val_dset = MulticomponentDataset(val_dsets)
        if len(test_data[0]) > 0:
            test_dsets = [
                make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
                for data in test_data
            ]
            test_dset = MulticomponentDataset(test_dsets)
        else:
            test_dset = None
    else:
        train_data = train_data[0]
        val_data = val_data[0]
        test_data = test_data[0]
        train_dset = make_dataset(train_data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
        val_dset = make_dataset(val_data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
        if len(test_data) > 0:
            test_dset = make_dataset(test_data, args.rxn_mode, args.multi_hot_atom_featurizer_mode)
        else:
            test_dset = None
    if args.task_type != "spectral":
        for dataset, label in zip(
            [train_dset, val_dset, test_dset], ["Training", "Validation", "Test"]
        ):
            column_headers, table_rows = summarize(args.target_columns, args.task_type, dataset)
            output = build_table(column_headers, table_rows, f"Summary of {label} Data")
            logger.info("\n" + output)

    return train_dset, val_dset, test_dset


def build_model(
    args,
    train_dset: MolGraphDataset | MulticomponentDataset,
    output_transform: UnscaleTransform,
    input_transforms: tuple[ScaleTransform, list[GraphTransform], list[ScaleTransform]],
) -> MPNN:
    mp_cls = AtomMessagePassing if args.atom_messages else BondMessagePassing

    X_d_transform, graph_transforms, V_d_transforms = input_transforms
    if isinstance(train_dset, MulticomponentDataset):
        mp_blocks = [
            mp_cls(
                train_dset.datasets[i].featurizer.atom_fdim,
                train_dset.datasets[i].featurizer.bond_fdim,
                d_h=args.message_hidden_dim,
                d_vd=(
                    train_dset.datasets[i].d_vd
                    if isinstance(train_dset.datasets[i], MoleculeDataset)
                    else 0
                ),
                bias=args.message_bias,
                depth=args.depth,
                undirected=args.undirected,
                dropout=args.dropout,
                activation=args.activation,
                V_d_transform=V_d_transforms[i],
                graph_transform=graph_transforms[i],
            )
            for i in range(train_dset.n_components)
        ]
        if args.mpn_shared:
            if args.reaction_columns is not None and args.smiles_columns is not None:
                raise ArgumentError(
                    argument=None,
                    message="Cannot use shared MPNN with both molecule and reaction data.",
                )

        mp_block = MulticomponentMessagePassing(mp_blocks, train_dset.n_components, args.mpn_shared)
        # NOTE(degraff): this if/else block should be handled by the init of MulticomponentMessagePassing
        # if args.mpn_shared:
        #     mp_block = MulticomponentMessagePassing(mp_blocks[0], n_components, args.mpn_shared)
        # else:
        d_xd = train_dset.datasets[0].d_xd
        n_tasks = train_dset.datasets[0].Y.shape[1]
        mpnn_cls = MulticomponentMPNN
    else:
        mp_block = mp_cls(
            train_dset.featurizer.atom_fdim,
            train_dset.featurizer.bond_fdim,
            d_h=args.message_hidden_dim,
            d_vd=train_dset.d_vd if isinstance(train_dset, MoleculeDataset) else 0,
            bias=args.message_bias,
            depth=args.depth,
            undirected=args.undirected,
            dropout=args.dropout,
            activation=args.activation,
            V_d_transform=V_d_transforms[0],
            graph_transform=graph_transforms[0],
        )
        d_xd = train_dset.d_xd
        n_tasks = train_dset.Y.shape[1]
        mpnn_cls = MPNN

    agg = Factory.build(AggregationRegistry[args.aggregation], norm=args.aggregation_norm)
    predictor_cls = PredictorRegistry[args.task_type]
    if args.loss_function is not None:
        task_weights = torch.ones(n_tasks) if args.task_weights is None else args.task_weights
        criterion = Factory.build(
            LossFunctionRegistry[args.loss_function],
            task_weights=task_weights,
            v_kl=args.v_kl,
            # threshold=args.threshold, TODO: Add in v2.1
            eps=args.eps,
        )
    else:
        criterion = None
    if args.metrics is not None:
        metrics = [Factory.build(MetricRegistry[metric]) for metric in args.metrics]
    else:
        metrics = None

    predictor = Factory.build(
        predictor_cls,
        input_dim=mp_block.output_dim + d_xd,
        n_tasks=n_tasks,
        hidden_dim=args.ffn_hidden_dim,
        n_layers=args.ffn_num_layers,
        dropout=args.dropout,
        activation=args.activation,
        criterion=criterion,
        n_classes=args.multiclass_num_classes,
        output_transform=output_transform,
        # spectral_activation=args.spectral_activation, TODO: Add in v2.1
    )

    if args.loss_function is None:
        logger.info(
            f"No loss function was specified! Using class default: {predictor_cls._T_default_criterion}"
        )

    return mpnn_cls(
        mp_block,
        agg,
        predictor,
        not args.no_batch_norm,
        metrics,
        args.warmup_epochs,
        args.init_lr,
        args.max_lr,
        args.final_lr,
        X_d_transform=X_d_transform,
    )


def train_model(
    args, train_loader, val_loader, test_loader, output_dir, output_transform, input_transforms
):
    if args.checkpoint is not None:
        model_paths = find_models(args.checkpoint)
        if args.ensemble_size != len(model_paths):
            logger.warning(
                f"The number of models in ensemble for each splitting of data is set to {len(model_paths)}."
            )
            args.ensemble_size = len(model_paths)

    for model_idx in range(args.ensemble_size):
        model_output_dir = output_dir / f"model_{model_idx}"
        model_output_dir.mkdir(exist_ok=True, parents=True)

        if args.pytorch_seed is None:
            seed = torch.seed()
            deterministic = False
        else:
            seed = args.pytorch_seed + model_idx
            deterministic = True

        torch.manual_seed(seed)

        if args.checkpoint or args.model_frzn is not None:
            mpnn_cls = (
                MulticomponentMPNN
                if isinstance(train_loader.dataset, MulticomponentDataset)
                else MPNN
            )
            model_path = model_paths[model_idx] if args.checkpoint else args.model_frzn
            model = mpnn_cls.load_from_file(model_path)

            if args.checkpoint:
                model.apply(
                    lambda m: setattr(m, "p", args.dropout)
                    if isinstance(m, torch.nn.Dropout)
                    else None
                )

            # TODO: model_frzn is deprecated and then remove in v2.2
            if args.model_frzn or args.freeze_encoder:
                model.message_passing.apply(lambda module: module.requires_grad_(False))
                model.message_passing.eval()
                model.bn.apply(lambda module: module.requires_grad_(False))
                model.bn.eval()
                for idx in range(args.frzn_ffn_layers):
                    model.predictor.ffn[idx].requires_grad_(False)
                    model.predictor.ffn[idx + 1].eval()
        else:
            model = build_model(args, train_loader.dataset, output_transform, input_transforms)
        logger.info(model)

        monitor_mode = "min" if model.metrics[0].minimize else "max"
        logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

        try:
            trainer_logger = TensorBoardLogger(model_output_dir, "trainer_logs")
        except ModuleNotFoundError as e:
            logger.warning(
                f"Unable to import TensorBoardLogger, reverting to CSVLogger (original error: {e})."
            )
            trainer_logger = CSVLogger(model_output_dir, "trainer_logs")

        if args.remove_checkpoints:
            temp_dir = TemporaryDirectory()
            checkpoint_dir = Path(temp_dir.name)
        else:
            checkpoint_dir = model_output_dir

        checkpointing = ModelCheckpoint(
            checkpoint_dir / "checkpoints",
            "best-{epoch}-{val_loss:.2f}",
            "val_loss",
            mode=monitor_mode,
            save_last=True,
        )

        if args.epochs != -1:
            patience = args.patience if args.patience is not None else args.epochs
            early_stopping = EarlyStopping("val_loss", patience=patience, mode=monitor_mode)
            callbacks = [checkpointing, early_stopping]
        else:
            callbacks = [checkpointing]

        trainer = pl.Trainer(
            logger=trainer_logger,
            enable_progress_bar=True,
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=args.epochs,
            callbacks=callbacks,
            gradient_clip_val=args.grad_clip,
            deterministic=deterministic,
        )
        trainer.fit(model, train_loader, val_loader)

        if test_loader is not None:
            if isinstance(trainer.strategy, DDPStrategy):
                torch.distributed.destroy_process_group()

                best_ckpt_path = trainer.checkpoint_callback.best_model_path
                trainer = pl.Trainer(
                    logger=trainer_logger,
                    enable_progress_bar=True,
                    accelerator=args.accelerator,
                    devices=1,
                )
                model = model.load_from_checkpoint(best_ckpt_path)
                predss = trainer.predict(model, dataloaders=test_loader)
            else:
                predss = trainer.predict(dataloaders=test_loader)

            preds = torch.concat(predss, 0)
            if model.predictor.n_targets > 1:
                preds = preds[..., 0]
            preds = preds.numpy()

            evaluate_and_save_predictions(
                preds, test_loader, model.metrics[:-1], model_output_dir, args
            )

        best_model_path = checkpointing.best_model_path
        model = model.__class__.load_from_checkpoint(best_model_path)
        p_model = model_output_dir / "best.pt"
        save_model(p_model, model, args.target_columns)
        logger.info(f"Best model saved to '{p_model}'")

        if args.remove_checkpoints:
            temp_dir.cleanup()


def evaluate_and_save_predictions(preds, test_loader, metrics, model_output_dir, args):
    if isinstance(test_loader.dataset, MulticomponentDataset):
        test_dset = test_loader.dataset.datasets[0]
    else:
        test_dset = test_loader.dataset
    targets = test_dset.Y
    mask = torch.from_numpy(np.isfinite(targets))
    targets = np.nan_to_num(targets, nan=0.0)
    weights = torch.ones(len(test_dset))
    lt_mask = torch.from_numpy(test_dset.lt_mask) if test_dset.lt_mask[0] is not None else None
    gt_mask = torch.from_numpy(test_dset.gt_mask) if test_dset.gt_mask[0] is not None else None

    individual_scores = dict()
    for metric in metrics:
        individual_scores[metric.alias] = []
        for i, col in enumerate(args.target_columns):
            if "multiclass" in args.task_type:
                preds_slice = torch.from_numpy(preds[:, i : i + 1, :])
                targets_slice = torch.from_numpy(targets[:, i : i + 1])
            else:
                preds_slice = torch.from_numpy(preds[:, i])
                targets_slice = torch.from_numpy(targets[:, i])
            preds_loss = metric(
                preds_slice,
                targets_slice,
                mask[:, i],
                weights,
                lt_mask[:, i] if lt_mask is not None else None,
                gt_mask[:, i] if gt_mask is not None else None,
            )
            individual_scores[metric.alias].append(preds_loss)

    logger.info("Entire Test Set results:")
    for metric in metrics:
        avg_loss = sum(individual_scores[metric.alias]) / len(individual_scores[metric.alias])
        logger.info(f"entire_test/{metric.alias}: {avg_loss}")

    if args.show_individual_scores:
        logger.info("Entire Test Set individual results:")
        for metric in metrics:
            for i, col in enumerate(args.target_columns):
                logger.info(
                    f"entire_test/{col}/{metric.alias}: {individual_scores[metric.alias][i]}"
                )

    names = test_loader.dataset.names
    if isinstance(test_loader.dataset, MulticomponentDataset):
        namess = list(zip(*names))
    else:
        namess = [names]

    columns = args.input_columns + args.target_columns
    if "multiclass" in args.task_type:
        columns = columns + [f"{col}_prob" for col in args.target_columns]
        formatted_probability_strings = np.apply_along_axis(
            lambda x: ",".join(map(str, x)), 2, preds
        )
        predicted_class_labels = preds.argmax(axis=-1)
        df_preds = pd.DataFrame(
            list(zip(*namess, *predicted_class_labels.T, *formatted_probability_strings.T)),
            columns=columns,
        )
    else:
        df_preds = pd.DataFrame(list(zip(*namess, *preds.T)), columns=columns)
    df_preds.to_csv(model_output_dir / "test_predictions.csv", index=False)


def main(args):
    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        target_cols=args.target_columns,
        ignore_cols=args.ignore_columns,
        splits_col=args.splits_column,
        weight_col=args.weight_column,
        bounded=args.loss_function is not None and "bounded" in args.loss_function,
    )

    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers, keep_h=args.keep_h, add_h=args.add_h
    )

    splits = build_splits(args, format_kwargs, featurization_kwargs)

    for fold_idx, (train_data, val_data, test_data) in enumerate(zip(*splits)):
        if args.num_folds == 1:
            output_dir = args.output_dir
        else:
            output_dir = args.output_dir / f"fold_{fold_idx}"

        output_dir.mkdir(exist_ok=True, parents=True)

        train_dset, val_dset, test_dset = build_datasets(args, train_data, val_data, test_data)

        if args.save_smiles_splits:
            save_smiles_splits(args, output_dir, train_dset, val_dset, test_dset)

        if args.checkpoint or args.model_frzn is not None:
            model_paths = find_models(args.checkpoint)
            if len(model_paths) > 1:
                logger.warning(
                    "Multiple checkpoint files were loaded, but only the scalers from "
                    f"{model_paths[0]} are used. It is assumed that all models provided have the "
                    "same data scalings, meaning they were trained on the same data."
                )
            model_path = model_paths[0] if args.checkpoint else args.model_frzn
            load_and_use_pretrained_model_scalers(model_path, train_dset, val_dset)
            input_transforms = (None, None, None)
            output_transform = None
        else:
            input_transforms = normalize_inputs(train_dset, val_dset, args)

            if "regression" in args.task_type:
                output_scaler = train_dset.normalize_targets()
                val_dset.normalize_targets(output_scaler)
                logger.info(
                    f"Train data: mean = {output_scaler.mean_} | std = {output_scaler.scale_}"
                )
                output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
            else:
                output_transform = None

        if not args.no_cache:
            train_dset.cache = True
            val_dset.cache = True

        train_loader = build_dataloader(
            train_dset,
            args.batch_size,
            args.num_workers,
            class_balance=args.class_balance,
            seed=args.data_seed,
        )
        if args.class_balance:
            logger.debug(
                f"With `--class-balance`, effective train size = {len(train_loader.sampler)}"
            )
        val_loader = build_dataloader(val_dset, args.batch_size, args.num_workers, shuffle=False)
        if test_dset is not None:
            test_loader = build_dataloader(
                test_dset, args.batch_size, args.num_workers, shuffle=False
            )
        else:
            test_loader = None

        train_model(
            args,
            train_loader,
            val_loader,
            test_loader,
            output_dir,
            output_transform,
            input_transforms,
        )


if __name__ == "__main__":
    # TODO: update this old code or remove it.
    parser = ArgumentParser()
    parser = TrainSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    TrainSubcommand.func(args)
