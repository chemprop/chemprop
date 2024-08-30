from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys
from typing import Iterator

from lightning import pytorch as pl
import numpy as np
import pandas as pd
import torch

from chemprop import data
from chemprop.cli.common import add_common_args, process_common_args, validate_common_args
from chemprop.cli.utils import LookupAction, Subcommand, build_data_from_files, make_dataset
from chemprop.models import load_model
from chemprop.nn.loss import LossFunctionRegistry
from chemprop.nn.predictors import MulticlassClassificationFFN, RegressionFFN
from chemprop.uncertainty import (
    UncertaintyCalibratorRegistry,
    UncertaintyEvaluatorRegistry,
    UncertaintyPredictorRegistry,
)
from chemprop.utils import Factory

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
        main(args)


def add_predict_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--test-path",
        required=True,
        type=Path,
        help="Path to an input CSV file containing SMILES",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--preds-path",
        type=Path,
        help="Specify path to which predictions will be saved. If the file extension is .pkl, it will be saved as a pickle file. Otherwise, chemprop will save predictions as a CSV. If multiple models are used to make predictions, the average predictions will be saved in the file, and another file ending in '_individual' with the same file extension will save the predictions for each individual model, with the column names being the target names appended with the model index (e.g., '_model_<index>').",
    )
    parser.add_argument(
        "--drop-extra-columns",
        action="store_true",
        help="Whether to drop all columns from the test data file besides the SMILES columns and the new prediction columns",
    )
    parser.add_argument(
        "--model-paths",
        "--model-path",
        required=True,
        type=Path,
        nargs="+",
        help="Location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, will recursively search and predict on all found (.pt) models.",
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        help="Column names to save the predictions to (by default, the predictions will be saved to columns named ``pred_0``, ``pred_1``, etc.)",
    )

    unc_args = parser.add_argument_group("Uncertainty and calibration args")
    unc_args.add_argument(
        "--cal-path", type=Path, help="Path to data file to be used for uncertainty calibration."
    )
    unc_args.add_argument(
        "--uncertainty-method",
        default="none",
        action=LookupAction(UncertaintyPredictorRegistry),
        help="The method of calculating uncertainty.",
    )
    unc_args.add_argument(
        "--calibration-method",
        action=LookupAction(UncertaintyCalibratorRegistry),
        help="The method used for calibrating the uncertainty calculated with uncertainty method.",
    )
    unc_args.add_argument(
        "--evaluation-methods",
        "--evaluation-method",
        nargs="+",
        action=LookupAction(UncertaintyEvaluatorRegistry),
        help="The methods used for evaluating the uncertainty performance if the test data provided includes targets. Available methods are [nll, miscalibration_area, ence, spearman] or any available classification or multiclass metric.",
    )
    # unc_args.add_argument(
    #     "--evaluation-scores-path", help="Location to save the results of uncertainty evaluations."
    # )
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
        help="Sets the percentile used in the calibration methods. Must be in the range (1, 100).",
    )
    unc_args.add_argument(
        "--regression-calibrator-metric",
        choices=["stdev", "interval"],
        help="Regression calibrators can output either a stdev or an inverval.",
    )
    unc_args.add_argument(
        "--cal-descriptors-path",
        nargs="+",
        action="append",
        help="Path to extra descriptors to concatenate to learned representation in calibration dataset.",
    )
    # TODO: Add in v2.1
    # unc_args.add_argument(
    #     "--calibration-phase-features-path",
    #     help=" ",
    # )
    unc_args.add_argument(
        "--cal-atom-features-path",
        nargs="+",
        action="append",
        help="Path to the extra atom features in calibration dataset.",
    )
    unc_args.add_argument(
        "--cal-atom-descriptors-path",
        nargs="+",
        action="append",
        help="Path to the extra atom descriptors in calibration dataset.",
    )
    unc_args.add_argument(
        "--cal-bond-features-path",
        nargs="+",
        action="append",
        help="Path to the extra bond descriptors in calibration dataset.",
    )

    return parser


def process_predict_args(args: Namespace) -> Namespace:
    if args.test_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None, message=f"Input data must be a CSV file. Got {args.test_path}"
        )
    if args.output is None:
        args.output = args.test_path.parent / (args.test_path.stem + "_preds.csv")
    if args.output.suffix not in [".csv", ".pkl"]:
        raise ArgumentError(
            argument=None, message=f"Output must be a CSV or Pickle file. Got {args.output}"
        )
    return args


def find_models(model_paths: list[Path]):
    collected_model_paths = []

    for model_path in model_paths:
        if model_path.suffix in [".ckpt", ".pt"]:
            collected_model_paths.append(model_path)
        elif model_path.is_dir():
            collected_model_paths.extend(list(model_path.rglob("*.pt")))
        else:
            raise ArgumentError(
                argument=None,
                message=f"Model path must be a .ckpt, .pt file, or a directory. Got {model_path}",
            )

    return collected_model_paths


def make_prediction_for_models(
    args: Namespace, model_paths: Iterator[Path], multicomponent: bool, output_path: Path
):
    model = load_model(model_paths[0], multicomponent)
    bounded = any(
        isinstance(model.criterion, LossFunctionRegistry[loss_function])
        for loss_function in LossFunctionRegistry.keys()
        if "bounded" in loss_function
    )
    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        target_cols=[],
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=bounded,
    )
    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers, keep_h=args.keep_h, add_h=args.add_h
    )

    test_data = build_data_from_files(
        args.test_path,
        **format_kwargs,
        p_descriptors=args.descriptors_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **featurization_kwargs,
    )
    logger.info(f"test size: {len(test_data[0])}")
    test_dsets = [
        make_dataset(d, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for d in test_data
    ]

    if multicomponent:
        test_dset = data.MulticomponentDataset(test_dsets)
    else:
        test_dset = test_dsets[0]

    test_loader = data.build_dataloader(test_dset, args.batch_size, args.num_workers, shuffle=False)

    if args.cal_path is not None:
        cal_data = build_data_from_files(
            args.cal_path,
            **format_kwargs,
            p_descriptors=args.cal_descriptors_path,
            p_atom_feats=args.cal_atom_features_path,
            p_bond_feats=args.cal_bond_features_path,
            p_atom_descs=args.cal_atom_descriptors_path,
            **featurization_kwargs,
        )
        logger.info(f"calibration size: {len(cal_data)}")
        cal_dsets = [
            make_dataset(d, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for d in cal_data
        ]
        if multicomponent:
            cal_dset = data.MulticomponentDataset(cal_dsets)
        else:
            cal_dset = test_dsets[0]
        cal_loader = data.build_dataloader(
            cal_dset, args.batch_size, args.num_workers, shuffle=False
        )
    else:
        cal_loader = None

    uncertinaty_predictor = Factory.build(UncertaintyPredictorRegistry[args.uncertainty_method])

    models = [load_model(model_path, multicomponent) for model_path in model_paths]

    trainer = pl.Trainer(
        logger=False, enable_progress_bar=True, accelerator=args.accelerator, devices=args.devices
    )

    test_individual_preds, test_uncs = uncertinaty_predictor(test_loader, models, trainer)
    test_preds = torch.mean(test_individual_preds, dim=0)

    if args.calibration_method is not None:
        uncertinaty_calibrator = Factory.build(
            UncertaintyCalibratorRegistry[args.calibration_method]
        )
        cal_targets = torch.from_numpy(cal_dset.Y)
        cal_mask = torch.from_numpy(np.isfinite(cal_targets))
        cal_individual_preds, cal_uncs = uncertinaty_predictor(cal_loader, models, trainer)
        cal_preds = torch.mean(cal_individual_preds, dim=0)
        if isinstance(model, RegressionFFN):
            uncertinaty_calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
        else:
            uncertinaty_calibrator.fit(None, cal_preds, cal_targets, cal_mask)
        test_uncs = uncertinaty_calibrator.apply(test_uncs)

    if args.evaluation_methods is not None:
        uncertinaty_evaluators = [
            Factory.build(UncertaintyEvaluatorRegistry[method])
            for method in args.evaluation_methods
        ]
        for evaluator in uncertinaty_evaluators:
            test_targets = torch.from_numpy(test_dset.Y)
            test_mask = torch.from_numpy(np.isfinite(test_targets))
            metric_value = evaluator.evaluate(test_preds, test_uncs, test_targets, test_mask)
            logger.info(
                f"Uncertainty evaluation metric: '{evaluator.alias}', metric value: {metric_value}"
            )

    if args.target_columns is not None:
        assert (
            len(args.target_columns) == model.n_tasks
        ), "Number of target columns must match the number of tasks."
        target_columns = args.target_columns
    else:
        target_columns = [
            f"pred_{i}" for i in range(test_preds.shape[1])
        ]  # TODO: need to improve this for cases like multi-task MVE and multi-task multiclass

    if isinstance(model.predictor, MulticlassClassificationFFN):
        target_columns = target_columns + [f"{col}_prob" for col in target_columns]
        predicted_class_labels = test_preds.argmax(axis=-1)
        formatted_probability_strings = np.apply_along_axis(
            lambda x: ",".join(map(str, x)), 2, test_preds
        )
        test_preds = np.concatenate(
            (predicted_class_labels, formatted_probability_strings), axis=-1
        )

    df_test = pd.read_csv(
        args.test_path, header=None if args.no_header_row else "infer", index_col=False
    )
    df_test[target_columns] = test_preds
    if output_path.suffix == ".pkl":
        df_test = df_test.reset_index(drop=True)
        df_test.to_pickle(output_path)
    else:
        df_test.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to '{output_path}'")

    if len(model_paths) > 1:
        target_columns = [
            f"{col}_model_{i}" for i in range(len(model_paths)) for col in target_columns
        ]

        if isinstance(model.predictor, MulticlassClassificationFFN):
            predicted_class_labels = test_individual_preds.argmax(axis=-1)
            formatted_probability_strings = np.apply_along_axis(
                lambda x: ",".join(map(str, x)), 3, test_individual_preds
            )
            test_individual_preds = np.concatenate(
                (predicted_class_labels, formatted_probability_strings), axis=-1
            )

        m, n, t = test_individual_preds.shape
        test_individual_preds = test_individual_preds.transpose(1, 0, 2).reshape(n, m * t)
        df_test = pd.read_csv(
            args.test_path, header=None if args.no_header_row else "infer", index_col=False
        )
        df_test[target_columns] = test_individual_preds

        output_path = output_path.parent / Path(
            str(args.output.stem) + "_individual" + str(output_path.suffix)
        )
        if output_path.suffix == ".pkl":
            df_test = df_test.reset_index(drop=True)
            df_test.to_pickle(output_path)
        else:
            df_test.to_csv(output_path, index=False)
        logger.info(f"Individual predictions saved to '{output_path}'")
        for i, model_path in enumerate(model_paths):
            logger.info(
                f"Results from model path {model_path} are saved under the column name ending with 'model_{i}'"
            )


def main(args):
    match (args.smiles_columns, args.reaction_columns):
        case [None, None]:
            n_components = 1
        case [_, None]:
            n_components = len(args.smiles_columns)
        case [None, _]:
            n_components = len(args.reaction_columns)
        case _:
            n_components = len(args.smiles_columns) + len(args.reaction_columns)

    multicomponent = n_components > 1

    model_paths = find_models(args.model_paths)

    make_prediction_for_models(args, model_paths, multicomponent, output_path=args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PredictSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    args = PredictSubcommand.func(args)
