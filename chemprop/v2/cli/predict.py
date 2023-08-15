from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
import warnings

from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from chemprop.v2 import data
from chemprop.v2.data.utils import split_data
from chemprop.v2.models import MetricRegistry, modules
from chemprop.v2.featurizers.utils import ReactionMode
from chemprop.v2.models.loss import LossFunction, build_loss

from chemprop.v2.cli.utils import Subcommand
from chemprop.v2.cli.utils_ import (
    build_data_from_files,
    get_mpnn_cls,
    make_dataset,
)

logger = logging.getLogger(__name__)


class PredictSubcommand(Subcommand):
    COMMAND = "predict"
    HELP = "use a pretrained chemprop model for prediction"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        return add_args(parser)
    
    @classmethod
    def func(cls, args: Namespace):
        process_args(args)
        main(args)


def add_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--input",
        "--data-path",
        help="path to an input CSV containing SMILES and associated target values",
    )
    parser.add_argument("-o", "--output-dir")
    parser.add_argument(
        "--logdir",
        nargs="?",
        const="chemprop_logs",
        help="runs will be logged to {logdir}/chemprop_{time}.log. If unspecified, will use 'output_dir'. If only the flag is given (i.e., '--logdir'), then will write to 'chemprop_logs'"
    )

    mp_args = parser.add_argument_group("message passing")
    mp_args.add_argument(
        "--message-hidden-dim", type=int, default=300, help="hidden dimension of the messages"
    )
    mp_args.add_argument(
        "--message-bias", action="store_true", help="add bias to the message passing layers"
    )
    mp_args.add_argument(
        "--depth", type=int, default=3, help="the number of message passing layers to stack"
    )
    mp_args.add_argument(
        "--undirected", action="store_true", help="pass messages on undirected bonds"
    )
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--activation", default="relu", help="activation function in message passing/FFN layers"
    )
    mp_args.add_argument(
        "--aggregation",
        "--agg",
        default="mean",
        choices=modules.ReadoutFactory.choices,
        help="aggregation mode to use during graph readout",
    )
    mp_args.add_argument(
        "--norm", type=float, default=100, help="normalization factor to use for 'norm' aggregation"
    )
    mp_args.add_argument(
        "--atom-messages", action="store_true", help="pass messages on atoms rather than bonds"
    )

    ffn_args = parser.add_argument_group("FFN args")
    ffn_args.add_argument(
        "--ffn-hidden-dim", type=int, default=300, help="hidden dimension in the FFN top model"
    )
    ffn_args.add_argument(
        "--ffn-num-layers", type=int, default=1, help="number of layers in FFN top model"
    )

    exta_mpnn_args = parser.add_argument_group("extra MPNN args")
    exta_mpnn_args.add_argument(
        "--multiclass-num-classes",
        type=int,
        help="the number of classes to predict in multiclass settings",
    )
    exta_mpnn_args.add_argument("--spectral-activation", default="exp", choices=["softplus", "exp"])

    data_args = parser.add_argument_group("input data parsing args")
    data_args.add_argument(
        "-d",
        "--dataset-type",
        default="regression",
        choices={l.split("-")[0] for l in LossFunction.registry.keys()}
    )
    data_args.add_argument(
        "--no-header-row", action="store_true", help="if there is no header in the input data CSV"
    )
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        type=int,
        default=[0],
        help="the columns in the CSV containing the SMILES strings of the inputs",
    )
    data_args.add_argument(
        "-t",
        "--target-columns",
        nargs="+",
        type=int,
        default=[1],
        help="the columns in the CSV containing the target values of the inputs",
    )

    data_args.add_argument(
        "--rxn-idxs",
        nargs="+",
        type=int,
        default=list(),
        help="the indices in the input SMILES containing reactions. Unless specified, each input is assumed to be a molecule. Should be a number in `[0, N)`, where `N` is the number of `--smiles-columns` specified",
    )
    data_args.add_argument("--cal-path")
    data_args.add_argument("--cal-features-path")
    data_args.add_argument("--cal-atom-features-path")
    data_args.add_argument("--cal-bond-features-path")
    data_args.add_argument("--cal-atom-descriptors-path")

    featurization_args = parser.add_argument_group("featurization args")
    featurization_args.add_argument("--rxn-mode", choices=ReactionMode.choices, default="reac_diff")
    featurization_args.add_argument(
        "--atom-features-path",
        help="the path to a .npy file containing a _list_ of `N` 2D arrays, where the `i`th array contains the atom features for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the atom features of the `j`th atom in the molecule.",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        help="the path to a .npy file containing a _list_ of `N` arrays, where the `i`th array contains the bond features for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the bond features of the `j`th bond in the molecule.",
    )
    featurization_args.add_argument(
        "--atom-descriptors-path",
        help="the path to a .npy file containing a _list_ of `N` arrays, where the `i`th array contains the atom descriptors for the `i`th molecule in the input data file. NOTE: each 2D array *must* have correct ordering with respect to the corresponding molecule in the data file. I.e., row `j` contains the atom descriptors of the `j`th atom in the molecule.",
    )
    featurization_args.add_argument("--features-generators", nargs="+")
    featurization_args.add_argument("--features-path")
    featurization_args.add_argument("--explicit-h", action="store_true")
    featurization_args.add_argument("--add-h", action="store_true")

    train_args = parser.add_argument_group("training args")
    train_args.add_argument("-b", "--batch-size", type=int, default=50)
    train_args.add_argument("--target-weights", type=float, nargs="+")
    train_args.add_argument(
        "-l", "--loss-function", choices={l.split("-")[1] for l in LossFunction.registry.keys()}
    )
    train_args.add_argument(
        "--v-kl", type=float, default=0.2, help="evidential/dirichlet regularization term weight"
    )
    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="evidential regularization epsilon"
    )
    train_args.add_argument("-T", "--threshold", type=float, help="spectral threshold limit")
    train_args.add_argument(
        "--metrics", nargs="+", choices=MetricRegistry.choices, help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing"
    )
    train_args.add_argument("-tw", "--task-weights", nargs="+", type=float, help="the weight to apply to an individual task in the overall loss")
    train_args.add_argument("--warmup-epochs", type=int, default=2)
    train_args.add_argument("--num-lrs", type=int, default=1)
    train_args.add_argument("--init-lr", type=float, default=1e-4)
    train_args.add_argument("--max-lr", type=float, default=1e-3)
    train_args.add_argument("--final-lr", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=30, help="the number of epochs to train over")

    parser.add_argument("--split", "--split-type", default="random")
    parser.add_argument("--split-sizes", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("-k", "--num-folds", type=int, default=1)
    parser.add_argument("--save-splits", action="store_true")

    parser.add_argument(
        "-g",
        "--n-gpu",
        type=int,
        default=1,
        help="the number of GPU(s) to use",
    )
    parser.add_argument(
        "-c",
        "--n-cpu",
        "--num-workers",
        type=int,
        default=0,
        help="the number of CPUs over which to parallelize data loading",
    )

    return parser


def process_args(args: Namespace):
    args.input = Path(args.input)
    if args.output is None:
        name = f"{args.input.stem}_preds.csv"
        args.output = Path(args.input.with_name(name))
    else:
        args.output = Path(args.output)
    args.logdir = Path(args.logdir or args.output_dir / "logs")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.logdir.mkdir(exist_ok=True, parents=True)


def validate_args(args):
    pass


def main(args):
    bond_messages = not args.atom_messages
    n_components = len(args.smiles_columns)
    n_tasks = len(args.target_columns)
    bounded = args.loss_function is not None and "bounded" in args.loss_function

    if n_components > 1:
        warnings.warn(
            "Multicomponent input is not supported at this time! Using only the 1st input..."
        )

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_columns=args.smiles_columns,
        bounded=bounded,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators,
        explicit_h=args.explicit_h,
        add_h=args.add_h,
        reaction=0 in args.rxn_idxs,
    )

    test_data = build_data_from_files(
        args.input,
        **format_kwargs,
        target_columns=[],
        p_features=args.features_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **featurization_kwargs,
    )
    logger.info(f"test size: {len(test_data)}")

    if args.cal_path is not None:
        cal_data = build_data_from_files(
            args.cal_path,
            **format_kwargs,
            target_columns=args.target_columns,
            p_features=args.cal_features_path,
            p_atom_feats=args.cal_atom_features_path,
            p_bond_feats=args.cal_bond_features_path,
            p_atom_descs=args.cal_atom_descriptors_path,
            **featurization_kwargs,
        )
        logger.info(f"calibration size: {len(cal_data)}")
    else:
        cal_data = None

    test_dset = make_dataset(test_data, bond_messages, args.rxn_mode)

    test_loader = data.MolGraphDataLoader(test_dset, args.batch_size, args.n_cpu)
    if cal_data is not None:
        cal_dset = make_dataset(cal_data, bond_messages, args.rxn_mode)
        cal_loader = data.MolGraphDataLoader(cal_dset, args.batch_size, args.n_cpu, shuffle=False)
    else:
        cal_loader = None

    mp_kwargs = dict(
        d_h=args.message_hidden_dim,
        bias=args.message_bias,
        depth=args.depth,
        undirected=args.undirected,
        dropout=args.dropout,
        activation=args.activation,
        aggregation=args.aggregation,
        norm=args.norm,
    )
    mp_block = modules.molecule_block(*test_dset.featurizer.shape, bond_messages, **mp_kwargs)

    extra_mpnn_kwargs = dict()
    if args.dataset_type == "multiclass":
        extra_mpnn_kwargs["n_classes"] = args.multiclass_num_classes
    elif args.dataset_type == "spectral":
        extra_mpnn_kwargs["spectral_activation"] = args.spectral_activation

    mpnn_cls = get_mpnn_cls(args.dataset_type, args.loss_function)
    model = mpnn_cls(
        mp_block,
        n_tasks,
        args.ffn_hidden_dim,
        args.ffn_num_layers,
        args.dropout,
        args.activation,
        **extra_mpnn_kwargs,
    )
    logger.info(model)
    
    trainer = pl.Trainer(
        logger=None,
        enable_progress_bar=True,
        accelerator="auto",
        devices=args.n_gpu if torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
    )
    
    predss = trainer.predict(model, test_loader)
    if cal_dset is not None:
        if args.dataset_type == "regression":
            model.loc, model.scale = float(scaler.mean_), float(scaler.scale_)
        predss_cal = trainer.predict(model, cal_loader)[0]


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    process_args(args)

    main(args)
