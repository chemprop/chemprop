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
    build_MAB_data_from_files,
    format_probability_string,
    make_dataset,
)
from chemprop.models.utils import load_model, load_output_columns
from chemprop.nn.metrics import BoundedMixin
from chemprop.nn.predictors import EvidentialFFN, MulticlassClassificationFFN, MveFFN, QuantileFFN
from chemprop.uncertainty import (
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
    HELP = "use a pretrained Chemprop model for prediction"

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
        help="Specify path to which predictions will be saved. If the file extension is .pkl, it will be saved as a pickle file. Otherwise, Chemprop will save predictions as a CSV. If multiple models are used to make predictions, the average predictions will be saved in the file, and another file ending in '_individual' with the same file extension will save the predictions for each individual model, with the column names being the target names appended with the model index (e.g., '_model_<index>').",
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
    unc_args.add_argument(
        "--cal-bond-descriptors-path",
        nargs="+",
        action="append",
        help="Path to the extra bond descriptors in calibration dataset.",
    )
    parser.add_argument(
        "--cal-constraints-path",
        help="Path to constraints applied to atomic/bond properties prediction for the calibration set.",
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
    args: Namespace,
    multicomponent: bool,
    mol_atom_bond: bool,
    is_calibration: bool,
    format_kwargs: dict,
):
    data_path = args.cal_path if is_calibration else args.test_path
    descriptors_path = args.cal_descriptors_path if is_calibration else args.descriptors_path
    atom_feats_path = args.cal_atom_features_path if is_calibration else args.atom_features_path
    bond_feats_path = args.cal_bond_features_path if is_calibration else args.bond_features_path
    atom_descs_path = (
        args.cal_atom_descriptors_path if is_calibration else args.atom_descriptors_path
    )
    bond_descs_path = (
        args.cal_bond_descriptors_path if is_calibration else args.bond_descriptors_path
    )
    constraints_path = args.cal_constraints_path if is_calibration else args.constraints_path

    featurization_kwargs = dict(
        molecule_featurizers=args.molecule_featurizers,
        keep_h=args.keep_h,
        add_h=args.add_h,
        ignore_stereo=args.ignore_stereo,
        reorder_atoms=args.reorder_atoms,
    )

    if mol_atom_bond:
        datas = build_MAB_data_from_files(
            data_path,
            **format_kwargs,
            p_descriptors=descriptors_path,
            descriptor_cols=args.descriptors_columns,
            p_atom_feats=atom_feats_path,
            p_bond_feats=bond_feats_path,
            p_atom_descs=atom_descs_path,
            p_bond_descs=bond_descs_path,
            p_constraints=constraints_path,
            constraints_cols_to_target_cols={
                col: i for i, col in enumerate(args.constraints_to_targets)
            }
            if args.constraints_to_targets is not None
            else None,
            n_workers=args.num_workers,
            **featurization_kwargs,
        )
    else:
        featurization_kwargs["use_cuikmolmaker_featurization"] = args.use_cuikmolmaker_featurization
        datas = build_data_from_files(
            data_path,
            **format_kwargs,
            p_descriptors=descriptors_path,
            descriptor_cols=args.descriptors_columns,
            p_atom_feats=atom_feats_path,
            p_bond_feats=bond_feats_path,
            p_atom_descs=atom_descs_path,
            n_workers=args.num_workers,
            **featurization_kwargs,
        )

    dsets = [
        make_dataset(
            d,
            args.rxn_mode,
            args.multi_hot_atom_featurizer_mode,
            args.use_cuikmolmaker_featurization,
            n_workers=args.num_workers,
        )
        for d in datas
    ]
    dset = data.MulticomponentDataset(dsets) if multicomponent else dsets[0]

    return data.build_dataloader(dset, args.batch_size, args.num_workers, shuffle=False)


def make_prediction_for_models(
    args: Namespace, model_paths: Iterator[Path], multicomponent: bool, output_path: Path
):
    mol_atom_bond = False
    models = [load_model(model_path, multicomponent) for model_path in model_paths]
    bounded = isinstance(models[0].criterion, BoundedMixin)
    output_columns = load_output_columns(model_paths[0])
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
    test_loader = prepare_data_loader(args, multicomponent, mol_atom_bond, False, format_kwargs)
    logger.info(f"test size: {len(test_loader.dataset)}")
    if args.cal_path is not None:
        format_kwargs["target_cols"] = output_columns
        cal_loader = prepare_data_loader(args, multicomponent, mol_atom_bond, True, format_kwargs)
        logger.info(f"calibration size: {len(cal_loader.dataset)}")

    uncertainty_estimator = Factory.build(
        UncertaintyEstimatorRegistry[args.uncertainty_method],
        ensemble_size=args.dropout_sampling_size,
        dropout=args.uncertainty_dropout_p,
    )

    trainer = pl.Trainer(
        logger=False, enable_progress_bar=True, accelerator=args.accelerator, devices=args.devices
    )
    test_individual_preds, test_individual_uncs = uncertainty_estimator(
        test_loader, models, trainer
    )

    test_preds = torch.mean(test_individual_preds, dim=0)
    if not isinstance(uncertainty_estimator, NoUncertaintyEstimator):
        test_uncs = torch.mean(test_individual_uncs, dim=0)
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
            test_uncs = uncertainty_calibrator.apply(test_individual_uncs)
        else:
            if isinstance(uncertainty_calibrator, RegressionCalibrator):
                uncertainty_calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
            else:
                uncertainty_calibrator.fit(cal_uncs, cal_targets, cal_mask)
            test_uncs = uncertainty_calibrator.apply(test_uncs)
            for i in range(test_individual_uncs.shape[0]):
                test_individual_uncs[i] = uncertainty_calibrator.apply(test_individual_uncs[i])

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
                metric_value = evaluator.evaluate(test_preds, test_uncs, test_targets, test_mask)
            else:
                metric_value = evaluator.evaluate(test_uncs, test_targets, test_mask)
            logger.info(f"{evaluator.alias}: {metric_value.tolist()}")

    if args.uncertainty_method == "none" and (
        isinstance(models[0].predictor, (MveFFN, EvidentialFFN, QuantileFFN))
    ):
        test_preds = test_preds[..., 0]
        test_individual_preds = test_individual_preds[..., 0]

    if output_columns is None:
        output_columns = [
            f"pred_{i}" for i in range(test_preds.shape[1])
        ]  # TODO: need to improve this for cases like multi-task MVE and multi-task multiclass

    save_predictions(args, models[0], output_columns, test_preds, test_uncs, output_path)

    if len(model_paths) > 1:
        save_individual_predictions(
            args,
            models[0],
            model_paths,
            output_columns,
            test_individual_preds,
            test_individual_uncs,
            output_path,
        )


def save_predictions(args, model, output_columns, test_preds, test_uncs, output_path):
    unc_columns = [f"{col}_unc" for col in output_columns]

    if isinstance(model.predictor, MulticlassClassificationFFN):
        output_columns = output_columns + [f"{col}_prob" for col in output_columns]
        predicted_class_labels = test_preds.argmax(axis=-1)
        formatted_probability_strings = format_probability_string(test_preds.numpy())
        test_preds = np.concatenate(
            (predicted_class_labels, formatted_probability_strings), axis=-1
        )

    df_test = pd.read_csv(
        args.test_path, header=None if args.no_header_row else "infer", index_col=False
    )
    df_test[output_columns] = test_preds

    if args.uncertainty_method not in ["none", "classification"]:
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
    test_individual_preds,
    test_individual_uncs,
    output_path,
):
    unc_columns = [
        f"{col}_unc_model_{i}" for i in range(len(model_paths)) for col in output_columns
    ]

    if isinstance(model.predictor, MulticlassClassificationFFN):
        output_columns = [
            item
            for i in range(len(model_paths))
            for col in output_columns
            for item in (f"{col}_model_{i}", f"{col}_prob_model_{i}")
        ]

        predicted_class_labels = test_individual_preds.argmax(axis=-1)
        formatted_probability_strings = format_probability_string(test_individual_preds.numpy())
        test_individual_preds = np.concatenate(
            (predicted_class_labels, formatted_probability_strings), axis=-1
        )
    else:
        output_columns = [
            f"{col}_model_{i}" for i in range(len(model_paths)) for col in output_columns
        ]

    m, n, t = test_individual_preds.shape
    test_individual_preds = np.transpose(test_individual_preds, (1, 0, 2)).reshape(n, m * t)
    df_test = pd.read_csv(
        args.test_path, header=None if args.no_header_row else "infer", index_col=False
    )
    df_test[output_columns] = test_individual_preds

    if args.uncertainty_method not in ["none", "classification", "ensemble"]:
        m, n, t = test_individual_uncs.shape
        test_individual_uncs = np.transpose(test_individual_uncs, (1, 0, 2)).reshape(n, m * t)
        df_test[unc_columns] = np.round(test_individual_uncs, 6)

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


def make_MAB_prediction_for_models(
    args: Namespace, model_paths: Iterator[Path], multicomponent: bool, output_path: Path
):
    mol_atom_bond = True
    models = [load_model(model_path, mol_atom_bond=True) for model_path in model_paths]
    bounded = isinstance(next(c for c in models[0].criterions if c is not None), BoundedMixin)

    mol_output_cols, atom_output_cols, bond_output_cols = load_output_columns(model_paths[0])

    format_kwargs = dict(
        smiles_cols=args.smiles_columns,
        mol_target_cols=mol_output_cols,
        atom_target_cols=atom_output_cols,
        bond_target_cols=bond_output_cols,
        weight_col=None,
        bounded=bounded,
    )
    test_loader = prepare_data_loader(args, multicomponent, mol_atom_bond, False, format_kwargs)
    logger.info(f"test size: {len(test_loader.dataset)}")
    if args.cal_path is not None:
        cal_loader = prepare_data_loader(args, multicomponent, mol_atom_bond, True, format_kwargs)
        logger.info(f"calibration size: {len(cal_loader.dataset)}")

    uncertainty_estimator = Factory.build(
        UncertaintyEstimatorRegistry[args.uncertainty_method],
        ensemble_size=args.dropout_sampling_size,
        dropout=args.uncertainty_dropout_p,
    )

    trainer = pl.Trainer(
        logger=False, enable_progress_bar=True, accelerator=args.accelerator, devices=args.devices
    )
    test_individual_predss, test_individual_uncss = uncertainty_estimator(
        test_loader, models, trainer
    )

    test_predss = [
        torch.mean(test_individual_preds, dim=0) if test_individual_preds is not None else None
        for test_individual_preds in test_individual_predss
    ]

    test_uncss = []
    cal_test_individual_uncss = []
    if args.calibration_method is not None:
        cal_individual_predss, cal_individual_uncss = uncertainty_estimator(
            cal_loader, models, trainer
        )
        cal_targetss = [
            cal_loader.dataset.Y,
            np.concatenate(cal_loader.dataset.atom_Y, axis=0),
            np.concatenate(cal_loader.dataset.bond_Y, axis=0),
        ]
        for test_individual_uncs, cal_individual_preds, cal_targets, cal_individual_uncs in zip(
            test_individual_uncss, cal_individual_predss, cal_targetss, cal_individual_uncss
        ):
            if test_individual_uncs is None:
                continue
            cal_mask = torch.from_numpy(np.isfinite(cal_targets))
            cal_targets = np.nan_to_num(cal_targets, nan=0.0)
            cal_targets = torch.from_numpy(cal_targets)
            cal_preds = torch.mean(cal_individual_preds, dim=0)
            cal_uncs = torch.mean(cal_individual_uncs, dim=0)

            uncertainty_calibrator = Factory.build(
                UncertaintyCalibratorRegistry[args.calibration_method],
                p=args.calibration_interval_percentile / 100,
                alpha=args.conformal_alpha,
            )

            if isinstance(uncertainty_calibrator, MVEWeightingCalibrator):
                uncertainty_calibrator.fit(cal_preds, cal_individual_uncs, cal_targets, cal_mask)
                test_uncss.append(uncertainty_calibrator.apply(test_individual_uncs))
            else:
                if isinstance(uncertainty_calibrator, RegressionCalibrator):
                    uncertainty_calibrator.fit(cal_preds, cal_uncs, cal_targets, cal_mask)
                else:
                    uncertainty_calibrator.fit(cal_uncs, cal_targets, cal_mask)
                for i in range(test_individual_uncs.shape[0]):
                    test_individual_uncs[i] = uncertainty_calibrator.apply(test_individual_uncs[i])
                cal_test_individual_uncss.append(test_individual_uncs)
                test_uncss.append(torch.mean(test_individual_uncs, dim=0))
        test_individual_uncss = cal_test_individual_uncss
    else:
        test_uncss = [
            torch.mean(test_individual_uncs, dim=0) if test_individual_uncs is not None else None
            for test_individual_uncs in test_individual_uncss
        ]

    if args.evaluation_methods is not None:
        uncertainty_evaluators = [
            Factory.build(UncertaintyEvaluatorRegistry[method])
            for method in args.evaluation_methods
        ]
        test_targetss = [
            test_loader.dataset.Y,
            np.concatenate(test_loader.dataset.atom_Y, axis=0),
            np.concatenate(test_loader.dataset.bond_Y, axis=0),
        ]
        test_maskss = [
            torch.from_numpy(np.isfinite(test_targets)) for test_targets in test_targetss
        ]
        test_targetss = [np.nan_to_num(test_targets, nan=0.0) for test_targets in test_targetss]
        test_targetss = [torch.from_numpy(test_targets) for test_targets in test_targetss]
        logger.info("Uncertainty evaluation metric:")
        for evaluator in uncertainty_evaluators:
            for test_preds, test_uncs, test_targets, test_mask, kind in zip(
                test_predss, test_uncss, test_targetss, test_maskss, ["mol", "atom", "bond"]
            ):
                if test_preds is None:
                    continue
                if isinstance(evaluator, RegressionEvaluator):
                    metric_value = evaluator.evaluate(
                        test_preds, test_uncs, test_targets, test_mask
                    )
                else:
                    metric_value = evaluator.evaluate(test_uncs, test_targets, test_mask)
                logger.info(f"{evaluator.alias} ({kind}): {metric_value.tolist()}")

    if args.uncertainty_method == "none" and (
        isinstance(next(p for p in models[0].predictors if p is not None), MveFFN)
        or isinstance(next(p for p in models[0].predictors if p is not None), EvidentialFFN)
    ):
        test_predss = [
            test_preds[..., 0] if test_preds is not None else None for test_preds in test_predss
        ]
        test_individual_predss = [
            test_individual_preds[..., 0] if test_individual_preds is not None else None
            for test_individual_preds in test_individual_predss
        ]

    output_columns = [
        col
        for cols in [mol_output_cols, atom_output_cols, bond_output_cols]
        if cols is not None
        for col in cols
    ]
    if len(output_columns) == 0:
        output_columns = [
            f"pred_{i}"
            for i in range(
                sum(test_preds.shape[1] for test_preds in test_predss if test_preds is not None)
            )
        ]

    atoms_per_molecule = [d.mol.GetNumAtoms() for d in test_loader.dataset.data]
    atom_split_indices = np.cumsum(atoms_per_molecule)[:-1]
    bonds_per_molecule = [d.mol.GetNumBonds() for d in test_loader.dataset.data]
    bond_split_indices = np.cumsum(bonds_per_molecule)[:-1]
    save_MAB_predictions(
        args,
        atom_split_indices,
        bond_split_indices,
        models[0],
        output_columns,
        test_predss,
        test_uncss,
        output_path,
    )

    if len(model_paths) > 1:
        for i in range(test_individual_predss[0].shape[0]):
            individual_output_path = output_path.with_name(
                output_path.stem + f"_model_{i}" + output_path.suffix
            )
            save_MAB_predictions(
                args,
                atom_split_indices,
                bond_split_indices,
                models[0],
                output_columns,
                test_individual_predss[i],
                test_individual_uncss[i],
                individual_output_path,
            )


def save_MAB_predictions(
    args,
    atom_split_indices,
    bond_split_indices,
    model,
    output_columns,
    test_predss,
    test_uncss,
    output_path,
):
    df_test = pd.read_csv(args.test_path, index_col=False)
    n_datapoints = len(df_test)

    unc_columns = [f"{col}_unc" for col in output_columns]

    mol_preds, atom_preds, bond_preds = test_predss
    mol_preds = mol_preds.numpy() if mol_preds is not None else None
    atom_preds = atom_preds.numpy() if atom_preds is not None else None
    bond_preds = bond_preds.numpy() if bond_preds is not None else None
    mol_uncs, atom_uncs, bond_uncs = test_uncss
    mol_uncs = mol_uncs.numpy() if mol_uncs is not None else None
    atom_uncs = atom_uncs.numpy() if atom_uncs is not None else None
    bond_uncs = bond_uncs.numpy() if bond_uncs is not None else None

    if isinstance(next(p for p in model.predictors if p is not None), MulticlassClassificationFFN):
        output_columns = output_columns + [f"{col}_prob" for col in output_columns]
        mols_class_probs = (
            format_probability_string(mol_preds) if mol_preds is not None else [None] * n_datapoints
        )
        atomss_class_probs = (
            np.split(format_probability_string(atom_preds), atom_split_indices)
            if atom_preds is not None
            else [None] * n_datapoints
        )
        bondss_class_probs = (
            np.split(format_probability_string(bond_preds), bond_split_indices)
            if bond_preds is not None
            else [None] * n_datapoints
        )

        mols_class_preds = (
            mol_preds.argmax(axis=-1) if mol_preds is not None else [None] * n_datapoints
        )
        atomss_class_preds = (
            np.split(atom_preds.argmax(axis=-1), atom_split_indices)
            if atom_preds is not None
            else [None] * n_datapoints
        )
        bondss_class_preds = (
            np.split(bond_preds.argmax(axis=-1), bond_split_indices)
            if bond_preds is not None
            else [None] * n_datapoints
        )

        outputs = [
            (
                *(mol_class_preds.tolist() if mol_class_preds is not None else []),
                *(atoms_class_preds.T.tolist() if atoms_class_preds is not None else []),
                *(bonds_class_preds.T.tolist() if bonds_class_preds is not None else []),
                *(mol_class_probs.tolist() if mol_class_probs is not None else []),
                *(atoms_class_probs.T.tolist() if atoms_class_probs is not None else []),
                *(bonds_class_probs.T.tolist() if bonds_class_probs is not None else []),
            )
            for mol_class_preds, atoms_class_preds, bonds_class_preds, mol_class_probs, atoms_class_probs, bonds_class_probs in zip(
                mols_class_preds,
                atomss_class_preds,
                bondss_class_preds,
                mols_class_probs,
                atomss_class_probs,
                bondss_class_probs,
            )
        ]
        df_preds = pd.DataFrame(outputs, columns=output_columns)
    else:
        mols_preds = mol_preds if mol_preds is not None else [None] * n_datapoints
        atomss_preds = (
            np.split(atom_preds, atom_split_indices)
            if atom_preds is not None
            else [None] * n_datapoints
        )
        bondss_preds = (
            np.split(bond_preds, bond_split_indices)
            if bond_preds is not None
            else [None] * n_datapoints
        )

        outputs = [
            (
                *(mol_preds.tolist() if mol_preds is not None else []),
                *(atoms_preds.T.tolist() if atoms_preds is not None else []),
                *(bonds_preds.T.tolist() if bonds_preds is not None else []),
            )
            for mol_preds, atoms_preds, bonds_preds in zip(mols_preds, atomss_preds, bondss_preds)
        ]
        df_preds = pd.DataFrame(outputs, columns=output_columns)

    df_test = pd.concat([df_test, df_preds], axis=1)

    if args.uncertainty_method not in ["none", "classification"]:
        mols_uncs = mol_uncs if mol_uncs is not None else [None] * n_datapoints
        atomss_uncs = (
            np.split(atom_uncs, atom_split_indices)
            if atom_uncs is not None
            else [None] * n_datapoints
        )
        bondss_uncs = (
            np.split(bond_uncs, bond_split_indices)
            if bond_uncs is not None
            else [None] * n_datapoints
        )

        outputs = [
            (
                *(mol_uncs.tolist() if mol_uncs is not None else []),
                *(
                    [list(atoms_unc) for atoms_unc in atoms_uncs.T]
                    if atoms_uncs is not None
                    else []
                ),
                *(
                    [list(bonds_unc) for bonds_unc in bonds_uncs.T]
                    if bonds_uncs is not None
                    else []
                ),
            )
            for mol_uncs, atoms_uncs, bonds_uncs in zip(mols_uncs, atomss_uncs, bondss_uncs)
        ]
        df_uncs = pd.DataFrame(outputs, columns=unc_columns)
        df_test = pd.concat([df_test, df_uncs], axis=1)

    if output_path.suffix == ".pkl":
        df_test = df_test.reset_index(drop=True)
        df_test.to_pickle(output_path)
    else:
        df_test.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to '{output_path}'")


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

    model_file = torch.load(model_paths[0], map_location=torch.device("cpu"), weights_only=False)
    mol_atom_bond = "atom_predictor" in model_file["hyper_parameters"].keys()

    if mol_atom_bond:
        make_MAB_prediction_for_models(args, model_paths, multicomponent, output_path=args.output)
    else:
        make_prediction_for_models(args, model_paths, multicomponent, output_path=args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = PredictSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    args = PredictSubcommand.func(args)
