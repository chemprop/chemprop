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
from chemprop.cli.common import (
    add_common_args,
    find_models,
    process_common_args,
    validate_common_args,
)
from chemprop.cli.utils import (
    LookupAction,
    Subcommand,
    build_data_from_files,
    build_mixed_data_from_files,
    make_dataset,
)
from chemprop.models.utils import load_mixed_model, load_model, load_output_columns
from chemprop.nn.metrics import LossFunctionRegistry
from chemprop.nn.predictors import EvidentialFFN, MulticlassClassificationFFN, MveFFN
from chemprop.uncertainty import (
    MixedNoUncertaintyEstimator,
    MVEWeightingCalibrator,
    NoUncertaintyEstimator,
    RegressionCalibrator,
    RegressionEvaluator,
    UncertaintyCalibratorRegistry,
    UncertaintyEstimatorRegistry,
    UncertaintyEvaluatorRegistry,
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

    unc_args = parser.add_argument_group("Uncertainty and calibration args")
    unc_args.add_argument(
        "--cal-path", type=Path, help="Path to data file to be used for uncertainty calibration."
    )
    unc_args.add_argument(
        "--uncertainty-method",
        default="none",
        action=LookupAction(UncertaintyEstimatorRegistry),
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
        "--conformal-alpha",
        type=float,
        default=0.1,
        help="Target error rate for conformal prediction. Must be in the range (0, 1).",
    )
    # TODO: Decide if we want to implment this in v2.1.x
    # unc_args.add_argument(
    #     "--regression-calibrator-metric",
    #     choices=["stdev", "interval"],
    #     help="Regression calibrators can output either a stdev or an inverval.",
    # )
    unc_args.add_argument(
        "--cal-descriptors-path",
        nargs="+",
        action="append",
        help="Path to extra descriptors to concatenate to learned representation in calibration dataset.",
    )
    # TODO: Add in v2.1.x
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


def prepare_data_loader(
    args: Namespace, multicomponent: bool, is_calibration: bool, format_kwargs: dict
):
    data_path = args.cal_path if is_calibration else args.test_path
    descriptors_path = args.cal_descriptors_path if is_calibration else args.descriptors_path
    atom_feats_path = args.cal_atom_features_path if is_calibration else args.atom_features_path
    bond_feats_path = args.cal_bond_features_path if is_calibration else args.bond_features_path
    atom_descs_path = (
        args.cal_atom_descriptors_path if is_calibration else args.atom_descriptors_path
    )

    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers,
        keep_h=args.keep_h,
        add_h=args.add_h,
        ignore_chirality=args.ignore_chirality,
    )
    if args.is_mixed:
        datas, mol_cols, atom_cols, bond_cols = build_mixed_data_from_files(
            data_path,
            **format_kwargs,
            p_descriptors=descriptors_path,
            p_atom_feats=atom_feats_path,
            p_bond_feats=bond_feats_path,
            p_atom_descs=atom_descs_path,
            **featurization_kwargs,
        )
        args.mixed_columns = [mol_cols, atom_cols, bond_cols]
    else:
        datas = build_data_from_files(
            data_path,
            **format_kwargs,
            p_descriptors=descriptors_path,
            p_atom_feats=atom_feats_path,
            p_bond_feats=bond_feats_path,
            p_atom_descs=atom_descs_path,
            **featurization_kwargs,
        )

    dsets = [make_dataset(d, args.rxn_mode, args.multi_hot_atom_featurizer_mode) for d in datas]
    dset = dsets[0]
    if multicomponent:
        dset = data.MulticomponentDataset(dsets)

    return data.build_dataloader(dset, args.batch_size, args.num_workers, shuffle=False)


def make_prediction_for_models(
    args: Namespace, model_paths: Iterator[Path], multicomponent: bool, output_path: Path
):
    if args.is_mixed:
        models = [load_mixed_model(model_path) for model_path in model_paths]
        args.uncertainty_method = "mixed_" + args.uncertainty_method
    else:
        models = [load_model(model_path, multicomponent) for model_path in model_paths]
    output_columns, mixed_columns = load_output_columns(model_paths[0])
    bounded = any(
        isinstance(models[0].criterion, LossFunctionRegistry[loss_function])
        for loss_function in LossFunctionRegistry.keys()
        if "bounded" in loss_function
    )
    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        ignore_cols=None,
        splits_col=None,
        weight_col=None,
        bounded=bounded,
    )
    format_kwargs["target_cols"] = output_columns if args.evaluation_methods is not None else []
    test_loader = prepare_data_loader(args, multicomponent, False, format_kwargs)

    logger.info(f"test size: {len(test_loader.dataset)}")
    if args.cal_path is not None:
        format_kwargs["target_cols"] = output_columns
        cal_loader = prepare_data_loader(args, multicomponent, True, format_kwargs)
        logger.info(f"calibration size: {len(cal_loader.dataset)}")

    uncertainty_estimator = Factory.build(
        UncertaintyEstimatorRegistry[args.uncertainty_method],
        ensemble_size=args.dropout_sampling_size,
        dropout=args.uncertainty_dropout_p,
    )  # make is_mixed one if args.is_mixed is true.

    trainer = pl.Trainer(
        logger=False, enable_progress_bar=True, accelerator=args.accelerator, devices=args.devices
    )
    test_individual_preds, test_individual_uncs = uncertainty_estimator(
        test_loader, models, trainer
    )
    if not args.is_mixed:
        test_individual_preds = [test_individual_preds]
        test_individual_uncs = [test_individual_uncs]
    test_preds = []
    test_uncs = []

    for idx, individual_pred in enumerate(test_individual_preds):
        test_preds.append(torch.mean(individual_pred, dim=0))

        if not isinstance(uncertainty_estimator, NoUncertaintyEstimator) and not isinstance(
            uncertainty_estimator, MixedNoUncertaintyEstimator
        ):
            test_uncs.append(torch.mean(test_individual_uncs[idx], dim=0))
        else:
            test_uncs = None

        if args.calibration_method is not None:
            uncertainty_calibrator = Factory.build(
                UncertaintyCalibratorRegistry[args.calibration_method],
                p=args.calibration_interval_percentile / 100,
                alpha=args.conformal_alpha,
            )
            cal_targets = cal_loader.dataset.Y
            cal_mask = torch.from_numpy(np.isfinite(cal_targets))
            cal_targets = np.nan_to_num(cal_targets, nan=0.0)
            cal_targets = torch.from_numpy(cal_targets)
            cal_individual_preds, cal_individual_uncs = uncertainty_estimator(
                cal_loader, models, trainer
            )
            cal_preds = torch.mean(cal_individual_preds, dim=0)
            cal_uncs = torch.mean(cal_individual_uncs, dim=0)
            if isinstance(uncertainty_calibrator, MVEWeightingCalibrator):
                uncertainty_calibrator.fit(cal_preds, cal_individual_uncs, cal_targets, cal_mask)
                test_uncs = uncertainty_calibrator.apply(cal_individual_uncs)
            else:
                if isinstance(uncertainty_calibrator, RegressionCalibrator):
                    uncertainty_calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
                else:
                    uncertainty_calibrator.fit(cal_uncs, cal_targets, cal_mask)
                test_uncs[idx] = uncertainty_calibrator.apply(test_uncs[idx])
                for i in range(test_individual_uncs[idx].shape[0]):
                    test_individual_uncs[idx][i] = uncertainty_calibrator.apply(
                        test_individual_uncs[idx][i]
                    )

        if args.evaluation_methods is not None:
            uncertainty_evaluators = [
                Factory.build(UncertaintyEvaluatorRegistry[method])
                for method in args.evaluation_methods
            ]
            logger.info("Uncertainty evaluation metric:")
            for evaluator in uncertainty_evaluators:
                test_targets = test_loader.dataset.Y
                test_mask = torch.from_numpy(np.isfinite(test_targets))
                test_targets = np.nan_to_num(test_targets, nan=0.0)
                test_targets = torch.from_numpy(test_targets)
                if isinstance(evaluator, RegressionEvaluator):
                    metric_value = evaluator.evaluate(
                        test_preds[idx], test_uncs[idx], test_targets, test_mask
                    )
                else:
                    metric_value = evaluator.evaluate(test_uncs[idx], test_targets, test_mask)
                logger.info(f"{evaluator.alias}: {metric_value.tolist()}")

        if args.uncertainty_method == "none":
            if args.is_mixed and (
                isinstance(models[0].predictors[0], MveFFN)
                or isinstance(models[0].predictors[0], EvidentialFFN)
            ):
                test_preds[idx] = test_preds[idx][..., 0]
                test_individual_preds[idx] = test_individual_preds[idx][..., 0]
            elif isinstance(models[0].predictor, MveFFN) or isinstance(
                models[0].predictor, EvidentialFFN
            ):
                test_preds[idx] = test_preds[idx][..., 0]
                test_individual_preds[idx] = test_individual_preds[idx][..., 0]

        if output_columns is None:
            output_columns = [
                f"pred_{i}" for i in range(test_preds[idx].shape[1])
            ]  # TODO: need to improve this for cases like multi-task MVE and multi-task multiclass

    test_preds = test_preds[0] if not args.is_mixed and test_preds is not None else test_preds
    test_uncs = test_uncs[0] if not args.is_mixed and test_uncs is not None else test_uncs

    save_predictions(
        args,
        models[0],
        output_columns,
        mixed_columns,
        test_preds,
        test_uncs,
        output_path,
        test_loader,
    )

    if len(model_paths) > 1:
        save_individual_predictions(
            args,
            models[0],
            model_paths,
            output_columns,
            mixed_columns,
            test_individual_preds,
            test_individual_uncs,
            output_path,
            test_loader,
        )


def save_predictions(
    args, model, output_columns, mixed_columns, test_preds, test_uncs, output_path, test_loader
):
    unc_columns = [f"{col}_unc" for col in output_columns]

    if (args.is_mixed and isinstance(model.predictors[0], MulticlassClassificationFFN)) or (
        not args.is_mixed and isinstance(model.predictor, MulticlassClassificationFFN)
    ):
        output_columns = output_columns + [f"{col}_prob" for col in output_columns]
        predicted_class_labels = test_preds.argmax(axis=-1)
        formatted_probability_strings = np.apply_along_axis(
            lambda x: ",".join(map(str, x)), 2, test_preds.numpy()
        )
        test_preds = np.concatenate(
            (predicted_class_labels, formatted_probability_strings), axis=-1
        )

    df_test = pd.read_csv(
        args.test_path, header=None if args.no_header_row else "infer", index_col=False
    )

    if args.is_mixed:
        mol_cols = mixed_columns[0]
        atom_cols = mixed_columns[1]
        bond_cols = mixed_columns[2]

        df_test.loc[:, mol_cols] = test_preds[0].tolist()

        for i in range(len(df_test)):
            atom_slices = test_loader.dataset._atom_slices
            if atom_slices is not None:
                first_atom = atom_slices.index(i)
                last_atom = first_atom + atom_slices.count(i)
                atom_preds = test_preds[1][first_atom:last_atom]
                df_test.loc[i, atom_cols] = [
                    str(atom_preds[:, j].tolist()) for j in range(len(atom_cols))
                ]

            bond_slices = test_loader.dataset._bond_slices
            if bond_slices is not None:
                first_bond = bond_slices.index(i)
                last_bond = first_bond + bond_slices.count(i)
                bond_preds = test_preds[2][first_bond:last_bond]
                df_test.loc[i, bond_cols] = [
                    str(bond_preds[:, j].tolist()) for j in range(len(bond_cols))
                ]
    else:
        df_test[output_columns] = test_preds[0]

    if args.uncertainty_method not in ["none", "mixed_none", "classification"]:
        df_test[unc_columns] = np.round(test_uncs, 6)

    if output_path.suffix == ".pkl":
        df_test = df_test.reset_index(drop=True)
        df_test.to_pickle(output_path)
    else:
        df_test.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to '{output_path}'")


def save_individual_predictions(
    args,
    model,
    model_paths,
    output_columns,
    mixed_columns,
    test_individual_preds,
    test_individual_uncs,
    output_path,
    test_loader,
):
    unc_columns = [
        f"{col}_unc_model_{i}" for i in range(len(model_paths)) for col in output_columns
    ]

    if (args.is_mixed and isinstance(model.predictors[0], MulticlassClassificationFFN)) or (
        not args.is_mixed and isinstance(model.predictor, MulticlassClassificationFFN)
    ):
        output_columns = [
            item
            for i in range(len(model_paths))
            for col in output_columns
            for item in (f"{col}_model_{i}", f"{col}_prob_model_{i}")
        ]

        predicted_class_labels = test_individual_preds[0].argmax(axis=-1)
        formatted_probability_strings = np.apply_along_axis(
            lambda x: ",".join(map(str, x)), 3, test_individual_preds[0].numpy()
        )
        test_individual_preds = [
            np.concatenate((predicted_class_labels, formatted_probability_strings), axis=-1)
        ]
    else:
        output_columns = [
            f"{col}_model_{i}" for i in range(len(model_paths)) for col in output_columns
        ]

    m, n, t = test_individual_preds[0].shape
    test_individual_preds[0] = np.transpose(test_individual_preds[0], (1, 0, 2)).reshape(
        n, m * t
    )  # TODO: edit for atoms/bonds
    df_test = pd.read_csv(
        args.test_path, header=None if args.no_header_row else "infer", index_col=False
    )

    if args.is_mixed:
        mol_cols = [f"{col}_model_{i}" for i in range(len(model_paths)) for col in mixed_columns[0]]
        atom_cols = [
            f"{col}_model_{i}" for i in range(len(model_paths)) for col in mixed_columns[1]
        ]
        bond_cols = [
            f"{col}_model_{i}" for i in range(len(model_paths)) for col in mixed_columns[2]
        ]

        df_test.loc[:, mol_cols] = test_individual_preds[0].tolist()

        for i in range(len(df_test)):
            atom_slices = test_loader.dataset._atom_slices
            if atom_slices is not None:
                first_atom = atom_slices.index(i)
                last_atom = first_atom + atom_slices.count(i)
                atom_preds = test_individual_preds[1][first_atom:last_atom]
                df_test.loc[i, atom_cols] = [
                    str(atom_preds[:, j].tolist()) for j in range(len(atom_cols))
                ]

            bond_slices = test_loader.dataset._bond_slices
            if bond_slices is not None:
                first_bond = bond_slices.index(i)
                last_bond = first_bond + bond_slices.count(i)
                bond_preds = test_individual_preds[2][first_bond:last_bond]
                df_test.loc[i, bond_cols] = [
                    str(bond_preds[:, j].tolist()) for j in range(len(bond_cols))
                ]
    else:
        df_test[output_columns] = test_individual_preds[0]

    if args.uncertainty_method not in ["none", "mixed_none", "classification"]:
        m, n, t = test_individual_uncs[0].shape
        test_individual_uncs[0] = np.transpose(test_individual_uncs[0], (1, 0, 2)).reshape(n, m * t)
        df_test[unc_columns] = np.round(test_individual_uncs[0], 6)

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
