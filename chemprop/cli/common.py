from argparse import ArgumentError, ArgumentParser, Namespace
import logging
from pathlib import Path

from chemprop.cli.utils import LookupAction
from chemprop.cli.utils.args import uppercase
from chemprop.featurizers import AtomFeatureMode, MoleculeFeaturizerRegistry, RxnMode
from chemprop.utils.utils import is_cuikmolmaker_available

logger = logging.getLogger(__name__)


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    data_args = parser.add_argument_group("Shared input data args")
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        nargs="+",
        help="Column names in the input CSV containing SMILES strings (uses the 0th column by default)",
    )
    data_args.add_argument(
        "-r",
        "--reaction-columns",
        nargs="+",
        help="Column names in the input CSV containing reaction SMILES in the format ``REACTANT>AGENT>PRODUCT``, where 'AGENT' is optional",
    )
    data_args.add_argument(
        "--no-header-row",
        action="store_true",
        help="Turn off using the first row in the input CSV as column names",
    )
    dataloader_args = parser.add_argument_group("Dataloader args")
    dataloader_args.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=0,
        help="""Number of workers for parallel data loading where 0 means sequential
(Warning: setting ``num_workers`` to a value greater than 0 can cause hangs on Windows and MacOS)""",
    )
    dataloader_args.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")

    parser.add_argument(
        "--accelerator", default="auto", help="Passed directly to the lightning ``Trainer()``"
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="Passed directly to the lightning ``Trainer()`` (must be a single string of comma separated devices, e.g. '1, 2' if specifying multiple devices)",
    )

    featurization_args = parser.add_argument_group("Featurization args")
    featurization_args.add_argument(
        "--rxn-mode",
        "--reaction-mode",
        type=uppercase,
        default="REAC_DIFF",
        choices=list(RxnMode.keys()),
        help="""Choices for construction of atom and bond features for reactions (case insensitive):

- ``REAC_PROD``: concatenates the reactants feature with the products feature
- ``REAC_DIFF``: concatenates the reactants feature with the difference in features between reactants and products (Default)
- ``PROD_DIFF``: concatenates the products feature with the difference in features between reactants and products
- ``REAC_PROD_BALANCE``: concatenates the reactants feature with the products feature, balances imbalanced reactions
- ``REAC_DIFF_BALANCE``: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions
- ``PROD_DIFF_BALANCE``: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions""",
    )
    # TODO: Update documenation for multi_hot_atom_featurizer_mode
    featurization_args.add_argument(
        "--multi-hot-atom-featurizer-mode",
        type=uppercase,
        default="V2",
        choices=list(AtomFeatureMode.keys()),
        help="""Choices for multi-hot atom featurization scheme. This will affect both non-reaction and reaction feturization (case insensitive):

- ``V1``: Corresponds to the original configuration employed in the Chemprop V1
- ``V2``: Tailored for a broad range of molecules, this configuration encompasses all elements in the first four rows of the periodic table, along with iodine. It is the default in Chemprop V2.
- ``ORGANIC``: This configuration is designed specifically for use with organic molecules for drug research and development and includes a subset of elements most common in organic chemistry, including H, B, C, N, O, F, Si, P, S, Cl, Br, and I.
- ``RIGR``: Modified V2 (default) featurizer using only the resonance-invariant atom and bond features.""",
    )
    featurization_args.add_argument(
        "--keep-h",
        action="store_true",
        help="Whether hydrogens explicitly specified in input should be kept in the mol graph",
    )
    featurization_args.add_argument(
        "--add-h", action="store_true", help="Whether hydrogens should be added to the mol graph"
    )
    data_args.add_argument(
        "--ignore-stereo",
        action="store_true",
        help="Ignore stereochemical information (R/S and Cis/Trans) in the input SMILES",
    )
    data_args.add_argument(
        "--reorder-atoms",
        action="store_true",
        help="Reorder atoms in the Chem.Mol object using the specified atom map numbers",
    )
    featurization_args.add_argument(
        "--molecule-featurizers",
        "--features-generators",
        nargs="+",
        action=LookupAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating molecule features to use as extra descriptors",
    )
    # TODO: add in v2.1 to deprecate features-generators and then remove in v2.2
    # featurization_args.add_argument(
    #     "--features-generators", nargs="+", help="Renamed to `--molecule-featurizers`."
    # )
    featurization_args.add_argument(
        "--descriptors-path",
        type=Path,
        help="Path to extra descriptors to concatenate to learned representation",
    )
    # TODO: Add in v2.1
    # featurization_args.add_argument(
    #     "--phase-features-path",
    #     help="Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.",
    # )
    featurization_args.add_argument(
        "--no-descriptor-scaling", action="store_true", help="Turn off extra descriptor scaling"
    )
    featurization_args.add_argument(
        "--no-atom-feature-scaling", action="store_true", help="Turn off extra atom feature scaling"
    )
    featurization_args.add_argument(
        "--no-atom-descriptor-scaling",
        action="store_true",
        help="Turn off extra atom descriptor scaling",
    )
    featurization_args.add_argument(
        "--no-bond-feature-scaling", action="store_true", help="Turn off extra bond feature scaling"
    )
    featurization_args.add_argument(
        "--no-bond-descriptor-scaling",
        action="store_true",
        help="Turn off extra bond descriptor scaling",
    )
    featurization_args.add_argument(
        "--atom-features-path",
        nargs="+",
        action="extend",
        help="If a single path is given, it is assumed to correspond to the 0-th molecule. Alternatively, it can be a two-tuple of molecule index and path to additional atom features to supply before message passing (e.g., ``--atom-features-path 0 /path/to/features_0.npz``) indicates that the features at the given path should be supplied to the 0-th component. To supply additional features for multiple components, repeat this argument on the command line for each component's respective values (e.g., ``--atom-features-path 0 path_zero --atom-features-path 1 path_one``) or pass each two-tuple in a series (e.g., ``--atom-features-path 0 path_zero 1 path_one``).",
    )
    featurization_args.add_argument(
        "--atom-descriptors-path",
        nargs="+",
        action="extend",
        help="If a single path is given, it is assumed to correspond to the 0-th molecule. Alternatively, it can be a two-tuple of molecule index and path to additional atom descriptors to supply after message passing (e.g., ``--atom-descriptors-path 0 /path/to/descriptors_0.npz`` indicates that the descriptors at the given path should be supplied to the 0-th component. To supply additional descriptors for multiple components, repeat this argument on the command line for each component's respective values (e.g., ``--atom-descriptors-path 0 path_zero --atom-descriptors-path 1 path_one``) or pass each two-tuple in a series (e.g., ``--atom-descriptors-path 0 path_zero 1 path_one``).",
    )
    featurization_args.add_argument(
        "--bond-features-path",
        nargs="+",
        action="extend",
        help="If a single path is given, it is assumed to correspond to the 0-th molecule. Alternatively, it can be a two-tuple of molecule index and path to additional bond features to supply before message passing (e.g., ``--bond-features-path 0 /path/to/features_0.npz`` indicates that the features at the given path should be supplied to the 0-th component. To supply additional features for multiple components, repeat this argument on the command line for each component's respective values (e.g., ``--bond-features-path 0 path_zero --bond-features-path 1 path_one``) or pass each two-tuple in a series (e.g., ``--bond-features-path 0 path_zero 1 path_one``).",
    )
    featurization_args.add_argument(
        "--bond-descriptors-path",
        nargs="+",
        action="extend",
        help="Path to additional bond descriptors to use with the learned bond representations after message passing. The file follows the same format as `--atom-descriptors-path`, i.e. the file is created using `np.savez('bond_descriptors.npz', *E_ds)` where `E_ds` is a list of 2D numpy arrays with shape `n_bonds x n_descriptors`.",
    )
    parser.add_argument(
        "--constraints-path",
        help="Path to a CSV file containing the constraints for atomic/bond properties prediction. The file should have one column for each property being constrained with no SMILES column. The order of the rows should match the order of the SMILES in the input CSV. See also `--constraints-to-targets` for how to specify which constraint applies to which prediction.",
    )
    parser.add_argument(
        "--constraints-to-targets",
        nargs="+",
        help="The column names of the atom or bond targets that correspond to each constraint column in the constraints CSV.",
    )
    featurization_args.add_argument(
        "--use-cuikmolmaker-featurization",
        action="store_true",
        help="Use ``cuik-molmaker`` package for accelerated atom and bond featurization.",
    )
    # TODO: Add in v2.2
    # parser.add_argument(
    #     "--constraints-path",
    #     help="Path to constraints applied to atomic/bond properties prediction.",
    # )

    return parser


def process_common_args(args: Namespace) -> Namespace:
    # TODO: add in v2.1 to deprecate features-generators and then remove in v2.2
    # if args.features_generators is not None:
    #     raise ArgumentError(
    #         argument=None,
    #         message="`--features-generators` has been renamed to `--molecule-featurizers`.",
    #     )

    # Bond descriptors are not supported for multi-component, but we treat it like atom descriptors
    for key in [
        "atom_features_path",
        "atom_descriptors_path",
        "bond_features_path",
        "bond_descriptors_path",
    ]:
        inds_paths = getattr(args, key)

        if not inds_paths:
            continue

        if len(inds_paths) == 1:
            setattr(args, key, {0: Path(inds_paths[0])})
            continue

        if len(inds_paths) % 2 != 0:
            raise ArgumentError(
                argument=None,
                message=f"Invalid argument list for --{key.replace('_', '-')}. It should be either a series of two-tuples of molecule index and a path, or a single path (assumed to be the 0-th molecule). Got {inds_paths}.",
            )

        try:
            inds = [int(ind) for ind in inds_paths[::2]]
        except ValueError:
            raise ArgumentError(
                argument=None,
                message=f"Invalid argument list for --{key.replace('_', '-')}. It should be either a series of two-tuples of molecule index and a path, or a single path (assumed to be the 0-th molecule). Got {inds_paths}.",
            )
        paths = [Path(path) for path in inds_paths[1::2]]

        if len(set(inds)) != len(inds):
            raise ArgumentError(
                argument=None,
                message=f"Duplicate --{key.replace('_', '-')} received for one of the molecules. Got {inds_paths}.",
            )

        setattr(args, key, {ind: path for ind, path in zip(inds, paths)})

    return args


def validate_common_args(args):
    # Validation that cuik-molmaker is installed if the flag is used
    if args.use_cuikmolmaker_featurization and not is_cuikmolmaker_available():
        raise ArgumentError(
            argument=None,
            message=f"cuik-molmaker is not installed. Please install it using `python {Path(__file__).parents[1] / Path('scripts/check_and_install_cuik_molmaker.py')}` before using the `--use-cuikmolmaker-featurization` flag.",
        )

    if args.use_cuikmolmaker_featurization:
        if args.keep_h:
            raise ArgumentError(
                argument=None,
                message="`--keep-h` is not supported when using cuik-molmaker featurization.",
            )
        if args.ignore_stereo:
            raise ArgumentError(
                argument=None,
                message="`--ignore-stereo` is not supported when using cuik-molmaker featurization.",
            )
        if args.reorder_atoms:
            raise ArgumentError(
                argument=None,
                message="`--reorder-atoms` is not supported when using cuik-molmaker featurization.",
            )


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
                message=f"Expected a .ckpt or .pt file, or a directory. Got {model_path}",
            )

    return collected_model_paths
