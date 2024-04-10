import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from chemprop.cli.utils import LookupAction
from chemprop.cli.utils.args import uppercase
from chemprop.featurizers import MoleculeFeaturizerRegistry, RxnMode, AtomFeatureMode

logger = logging.getLogger(__name__)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Path to a configuration file. If specified, overrides all other arguments.",
    )

    data_args = parser.add_argument_group("Shared input data args")
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        help="The column names in the input CSV containing SMILES strings. If unspecified, uses the the 0th column.",
    )
    data_args.add_argument(
        "-r",
        "--reaction-columns",
        nargs="+",
        help="The column names in the input CSV containing reaction SMILES in the format 'REACTANT>AGENT>PRODUCT', where 'AGENT' is optional.",
    )
    data_args.add_argument(
        "--no-header-row",
        action="store_true",
        help="If specified, the first row in the input CSV will not be used as column names.",
    )

    dataloader_args = parser.add_argument_group("Dataloader args")
    dataloader_args.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="""Number of workers for parallel data loading (0 means sequential).
Warning: setting num_workers>0 can cause hangs on Windows and MacOS.""",
    )
    dataloader_args.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size.")

    parser.add_argument(
        "--accelerator", default="auto", help="Passed directly to the lightning Trainer()."
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Passed directly to the lightning Trainer(). If specifying multiple devices, must be a single string of comma separated devices, e.g. '1, 2'.",
    )

    featurization_args = parser.add_argument_group("Featurization args")
    featurization_args.add_argument(
        "--rxn-mode",
        "--reaction-mode",
        type=uppercase,
        default="REAC_DIFF",
        choices=list(RxnMode.keys()),
        help="""Choices for construction of atom and bond features for reactions (case insensitive):
- 'reac_prod': concatenates the reactants feature with the products feature.
- 'reac_diff': concatenates the reactants feature with the difference in features between reactants and products. (Default)
- 'prod_diff': concatenates the products feature with the difference in features between reactants and products.
- 'reac_prod_balance': concatenates the reactants feature with the products feature, balances imbalanced reactions.
- 'reac_diff_balance': concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
- 'prod_diff_balance': concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.""",
    )
    # TODO: Update documenation for multi_hot_atom_featurizer_mode
    featurization_args.add_argument(
        "--multi-hot-atom-featurizer-mode",
        type=uppercase,
        default="V2",
        choices=list(AtomFeatureMode.keys()),
        help="""Choices for multi-hot atom featurization scheme. This will affect both non-reatction and reaction feturization (case insensitive):
- `V1`: Corresponds to the original configuration employed in the Chemprop V1.
- `V2`: Tailored for a broad range of molecules, this configuration encompasses all elements in the first four rows of the periodic table, along with iodine. It is the default in Chemprop V2.
- `ORGANIC`: Designed specifically for use with organic molecules for drug research and development, this configuration includes a subset of elements most common in organic chemistry, including H, B, C, N, O, F, Si, P, S, Cl, Br, and I.""",
    )
    featurization_args.add_argument(
        "--keep-h",
        action="store_true",
        help="Whether hydrogens explicitly specified in input should be kept in the mol graph.",
    )
    featurization_args.add_argument(
        "--add-h", action="store_true", help="Whether hydrogens should be added to the mol graph."
    )
    featurization_args.add_argument(
        "--features-generators",
        nargs="+",
        action=LookupAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating additional features.",
    )
    featurization_args.add_argument(
        "--descriptors-path",
        type=Path,
        help="Path to extra descriptors to concatenate to learned representation.",
    )
    # TODO: Add in v2.1
    # featurization_args.add_argument(
    #     "--phase-features-path",
    #     help="Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.",
    # )
    featurization_args.add_argument(
        "--no-descriptor-scaling", action="store_true", help="Turn off extra descriptor scaling."
    )
    featurization_args.add_argument(
        "--no-atom-feature-scaling",
        action="store_true",
        help="Turn off extra atom feature scaling.",
    )
    featurization_args.add_argument(
        "--no-atom-descriptor-scaling",
        action="store_true",
        help="Turn off extra atom descriptor scaling.",
    )
    featurization_args.add_argument(
        "--no-bond-feature-scaling",
        action="store_true",
        help="Turn off extra bond feature scaling.",
    )
    featurization_args.add_argument(
        "--atom-features-path",
        help="Path to the extra atom features. Used as atom features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--atom-descriptors-path",
        help="Path to the extra atom descriptors. Used as descriptors and concatenated to the machine learned atomic representation.",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        help="Path to the extra bond features. Used as bond features to featurize a given molecule.",
    )
    # TODO: Add in v2.2
    # parser.add_argument(
    #     "--constraints-path",
    #     help="Path to constraints applied to atomic/bond properties prediction.",
    # )

    return parser


def process_common_args(args: Namespace) -> Namespace:

    if args.config_path is None:
        return args

    with open(args.config_path, "r") as f:
        config_args = json.load(f)

    for key, value in config_args.items():
        setattr(args, key, value)

    if isinstance(args.config_path, str):
        args.config_path = Path(args.config_path)

    if isinstance(args.descriptors_path, str):
        args.descriptors_path = Path(args.descriptors_path)

    if isinstance(args.data_path, str):
        args.data_path = Path(args.data_path)

    if isinstance(args.output_dir, str):
        args.output_dir = Path(args.output_dir)

    return args


def validate_common_args(args):
    pass
