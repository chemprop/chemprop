from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
import pandas as pd

from lightning import pytorch as pl
import torch

from chemprop import data
from chemprop.featurizers import RxnMode
from chemprop.metrics import MetricRegistry
from chemprop.nn.agg import AggregationRegistry
from chemprop.nn.loss import LossFunction, LossFunctionRegistry
from chemprop.models import MPNN
from chemprop.nn.message_passing import BondMessagePassing

from chemprop.cli.utils import Subcommand, build_data_from_files, make_dataset
from chemprop.cli.common import add_common_args, process_common_args, validate_common_args


logger = logging.getLogger(__name__)


class PredictSubcommand(Subcommand):
    COMMAND = "predict"
    HELP = "use a pretrained chemprop model for prediction"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        return add_predict_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        validate_common_args(args)
        args = process_predict_args(args)
        validate_predict_args(args)
        main(args)


def add_predict_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i", "--test-path", required=True, help="Path to an input CSV file containing SMILES."
    )
    parser.add_argument(
        "-o",
        "--output",
        "--preds-path",
        help="Path to CSV or PICKLE file where predictions will be saved. If the file extension is .pkl, will be saved as a PICKLE file. If not provided and the test_path is /path/to/test/test.csv, predictions will be saved to /path/to/test/test_preds.csv.",
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
        choices=AggregationRegistry.choices,
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
        choices={l.split("-")[0] for l in LossFunction.registry.keys()},
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
    featurization_args.add_argument("--rxn-mode", choices=RxnMode.choices, default="reac_diff")
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
        "--metrics",
        nargs="+",
        choices=MetricRegistry.choices,
        help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing",
    )
    train_args.add_argument(
        "-tw",
        "--task-weights",
        nargs="+",
        type=float,
        help="the weight to apply to an individual task in the overall loss",
    )
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

    parser.add_argument("-g", "--n-gpu", type=int, default=1, help="the number of GPU(s) to use")
    parser.add_argument(
        "--drop-extra-columns",
        action="store_true",
        help="Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns.",
    )

    # TODO: add uncertainty and calibration
    # unc_args = parser.add_argument_group("uncertainty and calibration args")
    # unc_args.add_argument(
    #     "--ensemble-variance",
    #     type=None,
    #     help="Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path.",
    # )
    # unc_args.add_argument(
    #     "--individual-ensemble-predictions",
    #     type=bool,
    #     action="store_true",
    #     help="Whether to return the predictions made by each of the individual models rather than the average of the ensemble.",
    # )
    # unc_args.add_argument(
    #     "--uncertainty-method",
    #     #action=RegistryAction(TODO: make register for uncertainty methods)
    #     help="The method of calculating uncertainty.",
    # )
    # unc_args.add_argument(
    #     "--calibration-method",
    #     #action=RegistryAction(TODO: make register for calibration methods)
    #     help="Methods used for calibrating the uncertainty calculated with uncertainty method.",
    # )
    # unc_args.add_argument(
    #     "--evaluation-method",
    #     #action=RegistryAction(TODO: make register for evaluation methods)
    #     type=list[str],
    #     help="The methods used for evaluating the uncertainty performance if the test data provided includes targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric.",
    # )
    # unc_args.add_argument(
    #     "--evaluation-scores-path",
    #     help="Location to save the results of uncertainty evaluations.",
    # )
    # unc_args.add_argument(
    #     "--uncertainty-dropout-p",
    #     type=float,
    #     default=0.1,
    #     help="The probability to use for Monte Carlo dropout uncertainty estimation.",
    # )
    # unc_args.add_argument(
    #     "--dropout-sampling-size",
    #     type=int,
    #     default=10,
    #     help="The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training.",
    # )
    # unc_args.add_argument(
    #     "--calibration-interval-percentile",
    #     type=float,
    #     default=95,
    #     help="Sets the percentile used in the calibration methods. Must be in the range (1,100).",
    # )
    # unc_args.add_argument(
    #     "--regression-calibrator-metric",
    #     choices=['stdev', 'interval'],
    #     help="Regression calibrators can output either a stdev or an inverval.",
    # )
    # unc_args.add_argument(
    #     "--calibrationipath",
    #     help="Path to data file to be used for uncertainty calibration.",
    # )
    # unc_args.add_argument(
    #     "--calibration-features-path",
    #     type=list[str],
    #     help="Path to features data to be used with the uncertainty calibration dataset.",
    # )
    # unc_args.add_argument(
    #     "--calibration-phase-features-path",
    #     help=" ",
    # )
    # unc_args.add_argument(
    #     "--calibration-atom-descriptors-path",
    #     help="Path to the extra atom descriptors.",
    # )
    # unc_args.add_argument(
    #     "--calibration-bond-descriptors-path",
    #     help="Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.",
    # )

    return parser


def process_predict_args(args: Namespace) -> Namespace:
    args.separate_test_path = Path(args.separate_test_path)
    if args.output is None:
        name = f"{args.separate_test_path.stem}_preds.csv"
        args.output = Path(args.separate_test_path.with_name(name))
    else:
        args.output = Path(args.output)

    return args


def validate_predict_args(args):
    # TODO: once args.checkpoint_dir and args.checkpoint are consolidated, need to change this as well. Not able to make this required in common.py as it may not be provided for training.
    if args.checkpoint_path is None:
        raise ValueError("Must provide a checkpoint path for prediction.")


def main(args):
    model = MPNN.load_from_checkpoint(args.checkpoint_path)

    bond_messages = isinstance(model.message_passing, BondMessagePassing)
    bounded = any(
        isinstance(model.criterion, LossFunctionRegistry[loss_function])
        for loss_function in LossFunctionRegistry.keys()
        if "bounded" in loss_function
    )

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        bounded=bounded,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators, keep_h=args.keep_h, add_h=args.add_h
    )

    test_data = build_data_from_files(
        args.separate_test_path,
        **format_kwargs,
        target_cols=[],
        p_features=args.features_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **featurization_kwargs,
    )
    logger.info(f"test size: {len(test_data)}")

    # TODO: add uncertainty and calibration
    # if args.cal_path is not None:
    #     cal_data = build_data_from_files(
    #         args.cal_path,
    #         **format_kwargs,
    #         target_columns=args.target_columns,
    #         p_features=args.cal_features_path,
    #         p_atom_feats=args.cal_atom_features_path,
    #         p_bond_feats=args.cal_bond_features_path,
    #         p_atom_descs=args.cal_atom_descriptors_path,
    #         **featurization_kwargs,
    #     )
    #     logger.info(f"calibration size: {len(cal_data)}")
    # else:
    #     cal_data = None

    test_dset = make_dataset(test_data, bond_messages, args.rxn_mode)

    test_loader = data.MolGraphDataLoader(test_dset, args.batch_size, args.n_cpu, shuffle=False)
    # TODO: add uncertainty and calibration
    # if cal_data is not None:
    #     cal_dset = make_dataset(cal_data, bond_messages, args.rxn_mode)
    #     cal_loader = data.MolGraphDataLoader(cal_dset, args.batch_size, args.n_cpu, shuffle=False)
    # else:
    #     cal_loader = None

    logger.info(model)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="auto",
            devices=args.n_gpu if torch.cuda.is_available() else 1,
        )

        predss = trainer.predict(model, test_loader)

    # TODO: add uncertainty and calibration
    # if cal_dset is not None:
    #     if args.task_type == "regression":
    #         model.loc, model.scale = float(scaler.mean_), float(scaler.scale_)
    #     predss_cal = trainer.predict(model, cal_loader)[0]

    # TODO: might want to write a shared function for this as train.py might also want to do this.
    df_test = pd.read_csv(args.separate_test_path)
    preds = torch.concat(predss, 1).numpy()
    df_test[
        "preds"
    ] = preds.flatten()  # TODO: this will not work correctly for multi-target predictions
    if args.output.suffix == ".pkl":
        df_test.to_pickle(args.output, index=False)
    else:
        df_test.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PredictSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    args = PredictSubcommand.func(args)
