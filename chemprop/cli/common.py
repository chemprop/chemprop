from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

from chemprop.cli.utils import LookupAction
from chemprop.cli.utils.args import uppercase
from chemprop.featurizers import RxnMode, MoleculeFeaturizerRegistry

logger = logging.getLogger(__name__)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    data_args = parser.add_argument_group("shared input data args")
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
        help="The column names in the input CSV containing reaction SMILES in the format 'REACTANT>AGENT>PRODUCT'",
    )
    data_args.add_argument(
        "--no-header-row",
        action="store_true",
        help="If specified, the first row in the input CSV will not be used as column names.",
    )

    dataloader_args = parser.add_argument_group("dataloader args")
    dataloader_args.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the parallel data loading (0 means sequential).",
    )
    dataloader_args.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size.")

    # parser.add_argument(
    #     "--no-cuda", action="store_true", help="Turn off cuda (i.e., use CPU instead of GPU)."
    # )
    # parser.add_argument("--gpu", type=int, help="Which GPU to use.")
    parser.add_argument("-g", "--n-gpu", type=int, default=1, help="the number of GPU(s) to use")

    featurization_args = parser.add_argument_group("featurization args")
    featurization_args.add_argument(
        "--rxn-mode",
        "--reaction-mode",
        type=uppercase,
        default="REAC_DIFF",
        choices=list(RxnMode.keys()),
        help="""Choices for construction of atom and bond features for reactions (case insensitive):
- 'reac_prod': concatenates the reactants feature with the products feature.
- 'reac_diff': concatenates the reactants feature with the difference in features between reactants and products.
- 'prod_diff': concatenates the products feature with the difference in features between reactants and products.
- 'reac_prod_balance': concatenates the reactants feature with the products feature, balances imbalanced reactions.
- 'reac_diff_balance': concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
- 'prod_diff_balance': concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.""",
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
        action=LookupAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating additional features.",
    )
    featurization_args.add_argument(
        "--features-path",
        type=list[str], # TODO: why is this a list[str] instead of str?
        help="Path(s) to features to use in FNN (instead of features_generator).",
    )
    # TODO: Add in v2.1
    # featurization_args.add_argument(
    #     "--phase-features-path",
    #     help="Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.",
    # )
    featurization_args.add_argument(
        "--no-features-scaling", action="store_true", help="Turn off scaling of features."
    )
    featurization_args.add_argument(
        "--no-atom-descriptor-scaling", action="store_true", help="Turn off atom feature scaling."
    )
    featurization_args.add_argument(
        "--no-bond-descriptor-scaling", action="store_true", help="Turn off bond feature scaling."
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
        "--overwrite-default-atom-features",
        action="store_true",
        help="Overwrites the default atom descriptors with the new ones instead of concatenating them. Can only be used if atom_descriptors are used as a feature.",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        help="Path to the extra bond features. Used as bond features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--bond-descriptors-path",
        help="Path to the extra bond descriptors. Used as descriptors and concatenated to the machine learned bond representation.",
    )
    featurization_args.add_argument(
        "--overwrite-default-bond-features",
        action="store_true",
        help="Overwrites the default bond descriptors with the new ones instead of concatenating them. Can only be used if bond_descriptors are used as a feature.",
    )
    # TODO: Add in v2.2
    # parser.add_argument(
    #     "--constraints-path",
    #     help="Path to constraints applied to atomic/bond properties prediction.",
    # )

    return parser


def process_common_args(args: Namespace) -> Namespace:
    return args


def validate_common_args(args):
    pass
