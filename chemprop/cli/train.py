from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
import json
from copy import deepcopy
import pandas as pd
from rdkit import Chem

from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from chemprop.data import (
    MolGraphDataLoader,
    MolGraphDataset,
    MulticomponentDataset,
    ReactionDataset,
    MoleculeDataset,
)
from chemprop.data import SplitType, split_component
from chemprop.utils import Factory
from chemprop.models import MPNN, MulticomponentMPNN, save_model
from chemprop.nn import AggregationRegistry, LossFunctionRegistry, MetricRegistry, PredictorRegistry
from chemprop.nn.message_passing import (
    BondMessagePassing,
    AtomMessagePassing,
    MulticomponentMessagePassing,
)
from chemprop.nn.utils import Activation

from chemprop.cli.common import add_common_args, process_common_args, validate_common_args
from chemprop.cli.conf import NOW
from chemprop.cli.utils import (
    Subcommand,
    LookupAction,
    build_data_from_files,
    make_dataset,
    get_column_names,
)
from chemprop.cli.utils.args import uppercase

logger = logging.getLogger(__name__)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "train a chemprop model"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        return add_train_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        validate_common_args(args)
        args = process_train_args(args)
        validate_train_args(args)
        main(args)


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--data-path",
        required=True,
        type=Path,
        help="Path to an input CSV file containing SMILES and the associated target values.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--save-dir",
        type=Path,
        help="Directory where training outputs will be saved. Defaults to 'CURRENT_DIRECTORY/chemprop_training/STEM_OF_INPUT/TIME_STAMP'.",
    )

    # TODO: Add in v2.1
    # parser.add_argument(
    #     "--checkpoint-dir",
    #     help="Directory from which to load model checkpoints (walks directory and ensembles all models that are found).",
    # )
    # parser.add_argument("--checkpoint-path", help="Path to model checkpoint (:code:`.pt` file).")
    # parser.add_argument(
    #     "--checkpoint-paths",
    #     type=list[str],
    #     help="List of paths to model checkpoints (:code:`.pt` files).",
    # )
    # # TODO: Is this a prediction only argument?
    # parser.add_argument(
    #     "--checkpoint",
    #     help="Location of checkpoint(s) to use for ... If the location is a directory, chemprop walks it and ensembles all models that are found. If the location is a path or list of paths to model checkpoints (:code:`.pt` files), only those models will be loaded.",
    # )

    # TODO: Add in v2.1; see if we can tell lightning how often to log training loss
    # parser.add_argument(
    #     "--log-frequency",
    #     type=int,
    #     default=10,
    #     help="The number of batches between each logging of the training loss.",
    # )

    transfer_args = parser.add_argument_group("transfer learning args")
    transfer_args.add_argument(
        "--model-frzn",
        help="Path to model checkpoint file to be loaded for overwriting and freezing weights.",
    )
    transfer_args.add_argument(
        "--frzn-ffn-layers",
        type=int,
        default=0,
        help="Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn), where n is specified in the input. Automatically also freezes mpnn weights.",
    )
    # transfer_args.add_argument(
    #     "--freeze-first-only",
    #     action="store_true",
    #     help="Determines whether or not to use checkpoint_frzn for just the first encoder. Default (False) is to use the checkpoint to freeze all encoders. (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)",
    # )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Whether to save test split predictions during training.",
    )
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
        help="Number of models in ensemble for each splitting of data.",
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
        "--message-hidden-dim", type=int, default=300, help="hidden dimension of the messages"
    )
    mp_args.add_argument(
        "--message-bias", action="store_true", help="add bias to the message passing layers"
    )
    mp_args.add_argument("--depth", type=int, default=3, help="Number of message passing steps.")
    mp_args.add_argument(
        "--undirected",
        action="store_true",
        help="Pass messages on undirected bonds/edges (always sum the two relevant bond vectors).",
    )
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--mpn-shared",
        action="store_true",
        help="Whether to use the same message passing neural network for all input molecules. Only relevant if :code:`number_of_molecules > 1`",
    )
    mp_args.add_argument(
        "--activation",
        type=uppercase,
        default="RELU",
        choices=list(Activation.keys()),
        help="activation function in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--aggregation",
        "--agg",
        default="mean",
        action=LookupAction(AggregationRegistry),
        help="the aggregation mode to use during graph predictor",
    )
    mp_args.add_argument(
        "--aggregation-norm",
        type=float,
        default=100,
        help="normalization factor by which to divide summed up atomic features for 'norm' aggregation",
    )
    mp_args.add_argument(
        "--atom-messages", action="store_true", help="pass messages on atoms rather than bonds"
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
        "--ffn-hidden-dim", type=int, default=300, help="hidden dimension in the FFN top model"
    )
    ffn_args.add_argument(  # TODO: the default in v1 was 2. (see weights_ffn_num_layers option) Do we really want the default to now be 1?
        "--ffn-num-layers", type=int, default=1, help="number of layers in FFN top model"
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
        help="Don't use batch normalization after aggregation.",
    )
    extra_mpnn_args.add_argument(
        "--multiclass-num-classes",
        type=int,
        default=3,
        help="Number of classes when running multiclass classification.",
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
        help="the name of the column in the input CSV containg individual data weights",
    )
    train_data_args.add_argument(
        "--target-columns",
        nargs="+",
        help="Name of the columns containing target values. By default, uses all columns except the SMILES column and the :code:`ignore_columns`.",
    )
    train_data_args.add_argument(
        "--ignore-columns",
        nargs="+",
        help="Name of the columns to ignore when :code:`target_columns` is not provided.",
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
        help="Type of dataset. This determines the default loss function used during training. Defaults to regression.",
    )
    train_args.add_argument(
        "-l",
        "--loss-function",
        action=LookupAction(LossFunctionRegistry),
        help="Loss function to use during training. If not specified, will use the default loss function for the given task type (see documentation).",
    )
    train_args.add_argument(
        "--v-kl",
        "--evidential-regularization",
        type=float,
        default=0.0,
        help="Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model.",
    )

    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="evidential regularization epsilon"
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
        help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing",
    )
    # TODO: Add in v2.1
    # train_args.add_argument(
    #     "--show-individual-scores",
    #     action="store_true",
    #     help="Show all scores for individual targets, not just average, at the end.",
    # )
    train_args.add_argument(
        "--task-weights",
        nargs="+",
        type=float,
        help="the weight to apply to an individual task in the overall loss",
    )
    train_args.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`. Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.",
    )

    train_args.add_argument("--init-lr", type=float, default=1e-4, help="Initial learning rate.")
    train_args.add_argument("--max-lr", type=float, default=1e-3, help="Maximum learning rate.")
    train_args.add_argument("--final-lr", type=float, default=1e-4, help="Final learning rate.")
    train_args.add_argument(
        "--epochs", type=int, default=50, help="the number of epochs to train over"
    )
    train_args.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of epochs to wait for improvement before early stopping.",
    )
    train_args.add_argument(
        "--grad-clip",
        type=float,
        help="Passed directly to the lightning trainer which controls grad clipping. See the :code:`Trainer()` docstring for details.",
    )
    # TODO: Add in v2.1
    # train_args.add_argument(
    #     "--class-balance",
    #     action="store_true",
    #     help="Trains with an equal number of positives and negatives in each batch.",
    # )

    split_args = parser.add_argument_group("split args")
    split_args.add_argument(
        "--split",
        "--split-type",
        type=uppercase,
        default="RANDOM",
        choices=list(SplitType.keys()),
        help="Method of splitting the data into train/val/test (case insensitive).",
    )
    split_args.add_argument(
        "--split-sizes",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Split proportions for train/validation/test sets.",
    )
    split_args.add_argument(
        "--split-key-molecule",
        type=int,
        default=0,
        help="The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used (e.g., 'scaffold_balanced' or 'random_with_repeated_smiles'). Note that this index begins with zero for the first molecule.",
    )
    split_args.add_argument(
        "-k",
        "--num-folds",
        type=int,
        default=1,
        help="Number of folds when performing cross validation.",
    )
    # TODO: Add in v2.1
    # split_args.add_argument("--folds-file", help="Optional file of fold labels.")
    # split_args.add_argument(
    #     "--val-fold-index", type=int, help="Which fold to use as val for leave-one-out cross val."
    # )
    # split_args.add_argument(
    #     "--test-fold-index", type=int, help="Which fold to use as test for leave-one-out cross val."
    # )
    # split_args.add_argument(
    #     "--crossval-index-dir",
    #     help="Directory in which to find cross validation index files.",
    # )
    # split_args.add_argument(
    #     "--crossval-index-file",
    #     help="Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`.",
    # )
    split_args.add_argument(
        "--save-smiles-splits",
        action="store_true",
        help="Save smiles for each train/val/test splits for prediction convenience later.",
    )
    split_args.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets. When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed. Also used for shuffling data in :code:`MolGraphDataLoader` when :code:`shuffle` is True.",
    )
    
    parser.add_argument(
        "--pytorch-seed",
        type=int,
        default=0,
        help="Seed for PyTorch randomness (e.g., random initial weights).",
    )

    return parser


def process_train_args(args: Namespace) -> Namespace:
    if args.data_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None, message=f"Input data must be a CSV file. Got {args.data_path}"
        )
    if args.output_dir is None:
        args.output_dir = Path(f"chemprop_training/{args.data_path.stem}/{NOW}")
    args.output_dir.mkdir(exist_ok=True, parents=True)

    return args


def validate_train_args(args):
    pass


def normalize_inputs(train_dset, val_dset, args):
    X_d_scaler, V_f_scaler, E_f_scaler, V_d_scaler = None, None, None, None

    if isinstance(train_dset, MulticomponentDataset):
        d_xd = sum(dset.d_xd for dset in train_dset.datasets)
        d_vf = sum(
            0 if isinstance(dset, ReactionDataset) else dset.d_vf for dset in train_dset.datasets
        )
        d_ef = sum(
            0 if isinstance(dset, ReactionDataset) else dset.d_ef for dset in train_dset.datasets
        )
        d_vd = sum(
            0 if isinstance(dset, ReactionDataset) else dset.d_vd for dset in train_dset.datasets
        )
    else:
        d_xd = train_dset.d_xd
        d_vf = train_dset.d_vf if not isinstance(train_dset, ReactionDataset) else 0
        d_ef = train_dset.d_ef if not isinstance(train_dset, ReactionDataset) else 0
        d_vd = train_dset.d_vd if not isinstance(train_dset, ReactionDataset) else 0

    if d_xd > 0 and not args.no_descriptor_scaling:
        X_d_scaler = train_dset.normalize_inputs("X_d")
        val_dset.normalize_inputs("X_d", X_d_scaler)
        logger.info(f"Features: loc = {X_d_scaler.mean_}, scale = {X_d_scaler.scale_}")

    if d_vf > 0 and not args.no_atom_feature_scaling:
        V_f_scaler = train_dset.normalize_inputs("V_f")
        val_dset.normalize_inputs("V_f", V_f_scaler)
        logger.info(f"Atom features: loc = {V_f_scaler.mean_}, scale = {V_f_scaler.scale_}")

    if d_ef > 0 and not args.no_bond_feature_scaling:
        E_f_scaler = train_dset.normalize_inputs("E_f")
        val_dset.normalize_inputs("E_f", E_f_scaler)
        logger.info(f"Bond features: loc = {E_f_scaler.mean_}, scale = {E_f_scaler.scale_}")

    if d_vd > 0 and not args.no_atom_descriptor_scaling:
        V_d_scaler = train_dset.normalize_inputs("V_d")
        val_dset.normalize_inputs("V_d", V_d_scaler)
        logger.info(f"Atom descriptors: loc = {V_d_scaler.mean_}, scale = {V_d_scaler.scale_}")

    return X_d_scaler, V_f_scaler, E_f_scaler, V_d_scaler


def save_config(args: Namespace):
    command_config_path = args.output_dir / "config.json"
    with open(command_config_path, "w") as f:
        config = deepcopy(vars(args))
        for key in config:
            if isinstance(config[key], Path):
                config[key] = str(config[key])
        json.dump(config, f, indent=4)


def save_smiles_splits(args: Namespace, output_dir, train_dset, val_dset, test_dset):
    train_smis = train_dset.smiles
    df_train = pd.DataFrame(train_smis, columns=args.smiles_columns)
    df_train.to_csv(output_dir / "train_smiles.csv", index=False)

    val_smis = val_dset.smiles
    df_val = pd.DataFrame(val_smis, columns=args.smiles_columns)
    df_val.to_csv(output_dir / "val_smiles.csv", index=False)

    if test_dset is not None:
        test_smis = test_dset.smiles
        df_test = pd.DataFrame(test_smis, columns=args.smiles_columns)
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
    multicomponent = len(all_data) > 1

    split_kwargs = dict(sizes=args.split_sizes, seed=args.data_seed, num_folds=args.num_folds)
    split_kwargs["key_index"] = args.split_key_molecule if multicomponent else 0

    train_data, val_data, test_data = split_component(all_data, args.split, **split_kwargs)

    sizes = [len(train_data[0]), len(val_data[0]), len(test_data[0])]
    logger.info(f"train/val/test sizes: {sizes}")

    return train_data, val_data, test_data


def build_datasets(args, train_data, val_data, test_data):
    """build the train/val/test datasets, where :attr:`test_data` may be None"""
    multicomponent = len(train_data) > 1
    if multicomponent:
        train_dsets = [make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for data in train_data]
        val_dsets = [make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for data in val_data]
        train_dset = MulticomponentDataset(train_dsets)
        val_dset = MulticomponentDataset(val_dsets)
        if len(test_data[0]) > 0:
            test_dsets = [make_dataset(data, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for data in test_data]
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

    return train_dset, val_dset, test_dset


def build_model(args, train_dset: MolGraphDataset | MulticomponentDataset) -> MPNN:
    mp_cls = AtomMessagePassing if args.atom_messages else BondMessagePassing

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
        d_xd = sum(dset.d_xd for dset in train_dset.datasets)
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
        )
        d_xd = train_dset.d_xd
        n_tasks = train_dset.Y.shape[1]
        mpnn_cls = MPNN

    agg = Factory.build(AggregationRegistry[args.aggregation], norm=args.aggregation_norm)
    predictor_cls = PredictorRegistry[args.task_type]
    if args.loss_function is not None:
        criterion = Factory.build(
            LossFunctionRegistry[args.loss_function],
            v_kl=args.v_kl,
            # threshold=args.threshold, TODO: Add in v2.1
            eps=args.eps,
        )
    else:
        criterion = None

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
        # spectral_activation=args.spectral_activation, TODO: Add in v2.1
    )

    if args.loss_function is None:
        logger.info(
            f"No loss function was specified! Using class default: {predictor_cls._default_criterion}"
        )

    if args.model_frzn is not None:
        model = mpnn_cls.load_from_file(args.model_frzn)
        model.message_passing.apply(lambda module: module.requires_grad_(False))
        model.message_passing.apply(
            lambda m: setattr(m, "p", 0.0) if isinstance(m, torch.nn.Dropout) else None
        )
        model.bn.apply(lambda module: module.requires_grad_(False))
        for idx in range(args.frzn_ffn_layers):
            model.predictor.ffn[idx * 3].requires_grad_(False)
            setattr(model.predictor.ffn[idx * 3 + 2], "p", 0.0)

        return model

    metrics = [MetricRegistry[metric]() for metric in args.metrics] if args.metrics else None

    return mpnn_cls(
        mp_block,
        agg,
        predictor,
        not args.no_batch_norm,
        metrics,
        args.task_weights,
        args.warmup_epochs,
        args.init_lr,
        args.max_lr,
        args.final_lr,
    )


def train_model(args, train_loader, val_loader, test_loader, output_dir, scaler, input_scalers):
    for model_idx in range(args.ensemble_size):
        model_output_dir = output_dir / f"model_{model_idx}"
        model_output_dir.mkdir(exist_ok=True, parents=True)

        torch.manual_seed(args.pytorch_seed + model_idx)

        model = build_model(args, train_loader.dataset)
        logger.info(model)

        monitor_mode = "min" if model.metrics[0].minimize else "max"
        logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

        try:
            trainer_logger = TensorBoardLogger(model_output_dir, "trainer_logs")
        except ModuleNotFoundError:
            trainer_logger = CSVLogger(model_output_dir, "trainer_logs")

        checkpointing = ModelCheckpoint(
            model_output_dir / "checkpoints",
            "{epoch}-{val_loss:.2f}",
            "val_loss",
            mode=monitor_mode,
            save_last=True,
        )

        patience = args.patience if args.patience is not None else args.epochs
        early_stopping = EarlyStopping("val_loss", patience=patience, mode=monitor_mode)

        trainer = pl.Trainer(
            logger=trainer_logger,
            enable_progress_bar=True,
            accelerator="auto",
            devices=args.n_gpu if torch.cuda.is_available() else 1,
            max_epochs=args.epochs,
            callbacks=[checkpointing, early_stopping],
            gradient_clip_val=args.grad_clip,
        )
        trainer.fit(model, train_loader, val_loader)

        if test_loader is not None:
            if args.task_type == "regression":
                model.predictor.register_buffer("loc", torch.tensor(scaler.mean_).view(1, -1))
                model.predictor.register_buffer("scale", torch.tensor(scaler.scale_).view(1, -1))
            results = trainer.test(model, test_loader)[0]
            logger.info(f"Test results: {results}")

            if args.save_preds:
                predss = trainer.predict(model, test_loader)
                preds = torch.concat(predss, 0).numpy()
                columns = get_column_names(
                    args.data_path,
                    args.smiles_columns,
                    args.reaction_columns,
                    args.target_columns,
                    args.ignore_columns,
                    args.no_header_row,
                )
                df_preds = pd.DataFrame(
                    list(zip(test_loader.dataset.smiles, *preds.T)), columns=columns
                )
                df_preds.to_csv(model_output_dir / "test_predictions.csv", index=False)

        p_model = model_output_dir / "model.pt"
        output_scaler = scaler
        save_model(p_model, model, input_scalers, output_scaler)
        logger.info(f"Model saved to '{p_model}'")


def main(args):
    save_config(args)

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        target_cols=args.target_columns,
        ignore_cols=args.ignore_columns,
        weight_col=args.weight_column,
        bounded=args.loss_function is not None and "bounded" in args.loss_function,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators, keep_h=args.keep_h, add_h=args.add_h
    )

    no_cv = args.num_folds == 1
    train_data, val_data, test_data = build_splits(args, format_kwargs, featurization_kwargs)

    if no_cv:
        splits = ([train_data], [val_data], [test_data])
    else:
        splits = (train_data, val_data, test_data)

    for fold_idx, (train_data, val_data, test_data) in enumerate(zip(*splits)):
        if not no_cv:
            output_dir = args.output_dir / f"fold_{fold_idx}"
        else:
            output_dir = args.output_dir

        output_dir.mkdir(exist_ok=True, parents=True)

        train_dset, val_dset, test_dset = build_datasets(args, train_data, val_data, test_data)

        X_d_scaler, V_f_scaler, E_f_scaler, V_d_scaler = normalize_inputs(
            train_dset, val_dset, args
        )
        input_scalers = {"X_d": X_d_scaler, "V_f": V_f_scaler, "E_f": E_f_scaler, "V_d": V_d_scaler}

        if args.save_smiles_splits:
            save_smiles_splits(args, output_dir, train_dset, val_dset, test_dset)

        if "regression" in args.task_type:
            scaler = train_dset.normalize_targets()
            val_dset.normalize_targets(scaler)
            logger.info(f"Train data: mean = {scaler.mean_} | std = {scaler.scale_}")
        else:
            scaler = None

        train_loader = MolGraphDataLoader(
            train_dset, args.batch_size, args.num_workers, seed=args.data_seed
        )
        val_loader = MolGraphDataLoader(val_dset, args.batch_size, args.num_workers, shuffle=False)
        if test_dset is not None:
            test_loader = MolGraphDataLoader(
                test_dset, args.batch_size, args.num_workers, shuffle=False
            )
        else:
            test_loader = None

        train_model(args, train_loader, val_loader, test_loader, output_dir, scaler, input_scalers)


if __name__ == "__main__":
    # TODO: update this old code or remove it.
    parser = ArgumentParser()
    parser = TrainSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    TrainSubcommand.func(args)
