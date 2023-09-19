from argparse import ArgumentError, ArgumentParser, Namespace
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
from chemprop.v2.featurizers.reaction import RxnMode
from chemprop.v2.models.loss import LossFunction, LossFunctionRegistry
from chemprop.v2.models.model import MPNN
from chemprop.v2.models.modules.agg import AggregationRegistry

from chemprop.v2.cli.utils import Subcommand, RegistryAction
from chemprop.v2.cli.utils_ import build_data_from_files, get_mpnn_cls, make_dataset
from chemprop.v2.models.modules.message_passing.molecule import AtomMessageBlock, BondMessageBlock
from chemprop.v2.models.modules.readout import ReadoutRegistry, RegressionFFN
from chemprop.v2.utils.registry import Factory

from chemprop.v2.cli.train import add_common_args

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
        process_args(args)
        validate_args(args)
        main(args) 

def add_predict_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--input",
        "--test-path",
        dest="test_path",
        type=str,
        help="Path to CSV file containing SMILES for which predictions will be made.",
    )
    parser.add_argument(
        "-o", 
        "--output-dir",
        "--preds-path",
        dest="preds_path",
        type=str,
        help="Path to CSV or PICKLE file where predictions will be saved.",
    )
    parser.add_argument(
        "--drop-extra-columns",
        type=bool,
        action="store_true",
        help="Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns.",
    )

    if False: # to do: add uncertainty and calibration and delete this line
        unc_args = parser.add_argument_group("uncertainty and calibration args")
        unc_args.add_argument(
            "--ensemble-variance",
            type=None,
            help="Deprecated. Whether to calculate the variance of ensembles as a measure of epistemic uncertainty. If True, the variance is saved as an additional column for each target in the preds_path.",
        )
        unc_args.add_argument(
            "--individual-ensemble-predictions",
            type=bool,
            action="store_true",
            help="Whether to return the predictions made by each of the individual models rather than the average of the ensemble.",
        )
        unc_args.add_argument(
            "--uncertainty-method",
            #action=RegistryAction(to do: make register for uncertainty methods)
            help="The method of calculating uncertainty.",
        )
        unc_args.add_argument(
            "--calibration-method",
            #action=RegistryAction(to do: make register for calibration methods)
            help="Methods used for calibrating the uncertainty calculated with uncertainty method.",
        )
        unc_args.add_argument(
            "--evaluation-method",
            #action=RegistryAction(to do: make register for evaluation methods)
            type=list[str],
            help="The methods used for evaluating the uncertainty performance if the test data provided includes targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric.",
        )
        unc_args.add_argument(
            "--evaluation-scores-path",
            type=str,
            help="Location to save the results of uncertainty evaluations.",
        )
        unc_args.add_argument(
            "--uncertainty-dropout-p",
            type=float,
            default=0.1,
            help="The probability to use for Monte Carlo dropout uncertainty estimation.",
        )
        unc_args.add_argument(
            "--dropout-sampling-size",
            type=int,
            default=10,
            help="The number of samples to use for Monte Carlo dropout uncertainty estimation. Distinct from the dropout used during training.",
        )
        unc_args.add_argument(
            "--calibration-interval-percentile",
            type=float,
            default=95,
            help="Sets the percentile used in the calibration methods. Must be in the range (1,100).",
        )
        unc_args.add_argument(
            "--regression-calibrator-metric",
            choices=['stdev', 'interval'],
            help="Regression calibrators can output either a stdev or an inverval.",
        )
        unc_args.add_argument(
            "--calibrationipath",
            type=str,
            help="Path to data file to be used for uncertainty calibration.",
        )
        unc_args.add_argument(
            "--calibration-features-path",
            type=list[str],
            help="Path to features data to be used with the uncertainty calibration dataset.",
        )
        unc_args.add_argument(
            "--calibration-phase-features-path",
            type=str,
            help=" ",
        )
        unc_args.add_argument(
            "--calibration-atom-descriptors-path",
            type=str,
            help="Path to the extra atom descriptors.",
        )
        unc_args.add_argument(
            "--calibration-bond-descriptors-path",
            type=str,
            help="Path to the extra bond descriptors that will be used as bond features to featurize a given molecule.",
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
        no_header_row=args.no_header_row, smiles_columns=args.smiles_columns, bounded=bounded
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
