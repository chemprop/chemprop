from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from chemprop import data
from chemprop.cli.common import add_common_args, process_common_args, validate_common_args
from chemprop.cli.predict import find_models
from chemprop.cli.utils import (
    Subcommand,
    build_data_from_files,
    build_MAB_data_from_files,
    make_dataset,
)
from chemprop.models import load_model
from chemprop.nn.metrics import BoundedMixin

logger = logging.getLogger(__name__)


class FingerprintSubcommand(Subcommand):
    COMMAND = "fingerprint"
    HELP = "Use a pretrained Chemprop model to calculate learned representations."

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
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
            help="Specify the path where predictions will be saved. If the file extension is .npz, they will be saved as a npz file. Otherwise, the predictions will be saved as a CSV. The index of the model will be appended to the filename's stem. By default, predictions will be saved to the same location as ``--test-path`` with '_fps' appended (e.g., 'PATH/TO/TEST_PATH_fps_0.csv').",
        )
        parser.add_argument(
            "--model-paths",
            "--model-path",
            required=True,
            type=Path,
            nargs="+",
            help="Specify location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, chemprop will recursively search and predict on all found (.pt) models.",
        )
        parser.add_argument(
            "--ffn-block-index",
            required=True,
            type=int,
            default=-1,
            help="The index indicates which linear layer returns the encoding in the FFN. An index of 0 denotes the post-aggregation representation through a 0-layer MLP, while an index of 1 represents the output from the first linear layer in the FFN, and so forth.",
        )

        return parser

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        validate_common_args(args)
        args = process_fingerprint_args(args)
        main(args)


def process_fingerprint_args(args: Namespace) -> Namespace:
    if args.test_path.suffix not in [".csv"]:
        raise ArgumentError(
            argument=None, message=f"Input data must be a CSV file. Got {args.test_path}"
        )
    if args.output is None:
        args.output = args.test_path.parent / (args.test_path.stem + "_fps.csv")
    if args.output.suffix not in [".csv", ".npz"]:
        raise ArgumentError(
            argument=None, message=f"Output must be a CSV or NPZ file. Got '{args.output}'."
        )
    return args


def make_fingerprint_for_model(
    args: Namespace, model_path: Path, multicomponent: bool, mol_atom_bond: bool, output_path: Path
):
    model = load_model(model_path, multicomponent, mol_atom_bond)
    model.eval()

    criterion = (
        model.criterion if not mol_atom_bond else next(c for c in model.criterions if c is not None)
    )
    bounded = isinstance(criterion, BoundedMixin)

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
        molecule_featurizers=args.molecule_featurizers,
        keep_h=args.keep_h,
        add_h=args.add_h,
        ignore_stereo=args.ignore_stereo,
        reorder_atoms=args.reorder_atoms,
    )

    if mol_atom_bond:
        for key in ["no_header_row", "rxn_cols", "ignore_cols", "splits_col", "target_cols"]:
            format_kwargs.pop(key, None)
        test_data = build_MAB_data_from_files(
            args.test_path,
            **format_kwargs,
            mol_target_cols=None,
            atom_target_cols=None,
            bond_target_cols=None,
            p_descriptors=args.descriptors_path,
            descriptor_cols=args.descriptors_columns,
            p_atom_feats=args.atom_features_path,
            p_bond_feats=args.bond_features_path,
            p_atom_descs=args.atom_descriptors_path,
            p_bond_descs=args.bond_descriptors_path,
            p_constraints=None,
            constraints_cols_to_target_cols=None,
            **featurization_kwargs,
        )
    else:
        featurization_kwargs["use_cuikmolmaker_featurization"] = args.use_cuikmolmaker_featurization
        test_data = build_data_from_files(
            args.test_path,
            **format_kwargs,
            p_descriptors=args.descriptors_path,
            descriptor_cols=args.descriptors_columns,
            p_atom_feats=args.atom_features_path,
            p_bond_feats=args.bond_features_path,
            p_atom_descs=args.atom_descriptors_path,
            **featurization_kwargs,
        )
    logger.info(f"test size: {len(test_data[0])}")

    test_dsets = [
        make_dataset(
            d,
            args.rxn_mode,
            args.multi_hot_atom_featurizer_mode,
            args.use_cuikmolmaker_featurization,
            n_workers=args.num_workers,
        )
        for d in test_data
    ]

    if multicomponent:
        test_dset = data.MulticomponentDataset(test_dsets)
    else:
        test_dset = test_dsets[0]

    test_loader = data.build_dataloader(test_dset, args.batch_size, args.num_workers, shuffle=False)

    logger.info(model)

    with torch.no_grad():
        if not mol_atom_bond:
            if multicomponent:
                encodings = [
                    model.encoding(batch.bmgs, batch.V_ds, batch.X_d, args.ffn_block_index)
                    for batch in test_loader
                ]
            else:
                encodings = [
                    model.encoding(batch.bmg, batch.V_d, batch.X_d, args.ffn_block_index)
                    for batch in test_loader
                ]
            H = torch.cat(encodings, 0).numpy()

            if output_path.suffix in [".npz"]:
                np.savez(output_path, H=H)
            elif output_path.suffix == ".csv":
                fingerprint_columns = [f"fp_{i}" for i in range(H.shape[1])]
                df_fingerprints = pd.DataFrame(H, columns=fingerprint_columns)
                df_fingerprints.to_csv(output_path, index=False)
            else:
                raise ArgumentError(
                    argument=None, message=f"Output must be a CSV or npz file. Got {args.output}."
                )
        else:
            encodings = [
                model.encoding(batch.bmg, batch.V_d, batch.E_d, batch.X_d, args.ffn_block_index)
                for batch in test_loader
            ]
            Hs = [
                torch.cat(encoding, 0).numpy() if encoding[0] is not None else None
                for encoding in zip(*encodings)
            ]
            for H, kind in zip(Hs, ["mol", "atom", "bond"]):
                if H is None:
                    continue
                match output_path.suffix:
                    case ".npz":
                        np.savez(
                            output_path.with_stem(output_path.stem + f"_{kind}_fingerprints"), H=H
                        )
                    case ".csv":
                        fingerprint_columns = [f"fp_{i}" for i in range(H.shape[1])]
                        df_fingerprints = pd.DataFrame(H, columns=fingerprint_columns)
                        df_fingerprints.to_csv(
                            output_path.with_stem(output_path.stem + f"_{kind}_fingerprints"),
                            index=False,
                        )
                    case _:
                        raise ArgumentError(
                            argument=None,
                            message=f"Output must be a CSV or npz file. Got {args.output}.",
                        )
        logger.info(f"Fingerprints saved to '{output_path}'")


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

    for i, model_path in enumerate(model_paths):
        logger.info(f"Fingerprints with model {i} at '{model_path}'")
        output_path = args.output.parent / f"{args.output.stem}_{i}{args.output.suffix}"
        make_fingerprint_for_model(args, model_path, multicomponent, mol_atom_bond, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = FingerprintSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    args = FingerprintSubcommand.func(args)
