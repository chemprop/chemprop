from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from lightning import pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import torch

from chemprop.v2 import data
from chemprop.v2.data.utils import split_data
from chemprop.v2.featurizers import RxnMode
from chemprop.v2.nn.loss import LossFunction, LossFunctionRegistry
from chemprop.v2.models import MPNN
from chemprop.v2.nn.agg import AggregationRegistry
from chemprop.v2.nn.message_passing import AtomMessageBlock, BondMessageBlock
from chemprop.v2.nn.readout import ReadoutRegistry, RegressionFFN
from chemprop.v2.utils import Factory

from chemprop.v2.cli.utils import Subcommand, LookupAction
from chemprop.v2.cli.utils_ import build_data_from_files, get_mpnn_cls, make_dataset

from chemprop.v2.cli.common import add_common_args, process_common_args, validate_common_args

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

    bond_messages = isinstance(model.message_passing, BondMessageBlock)
    bounded = any(
        isinstance(model.criterion, LossFunctionRegistry[loss_function])
        for loss_function in LossFunctionRegistry.keys()
        if "bounded" in loss_function
    )

    format_kwargs = dict(
        no_header_row=args.no_header_row, smiles_columns=args.smiles_columns, bounded=bounded
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators,
        keep_h=args.keep_h,
        add_h=args.add_h,
        reaction=0 in args.rxn_idxs,
    )

    test_data = build_data_from_files(
        args.separate_test_path,
        **format_kwargs,
        target_columns=[],
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
