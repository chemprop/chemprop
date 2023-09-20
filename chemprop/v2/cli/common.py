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
from chemprop.v2.models import MetricRegistry
from chemprop.v2.featurizers.reaction import RxnMode
from chemprop.v2.models.loss import LossFunctionRegistry
from chemprop.v2.models.model import MPNN
from chemprop.v2.models.modules.agg import AggregationRegistry
from chemprop.v2.featurizers.featurizers import MoleculeFeaturizerRegistry

from chemprop.v2.cli.utils import Subcommand, RegistryAction
from chemprop.v2.cli.utils_ import build_data_from_files, make_dataset
from chemprop.v2.models.modules.message_passing.molecule import AtomMessageBlock, BondMessageBlock
from chemprop.v2.models.modules.readout import ReadoutRegistry, RegressionFFN
from chemprop.v2.utils.registry import Factory

logger = logging.getLogger(__name__)

def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-i",
        "--input",
        "--test-path", # v1 option for predicting
        "--data-path", # v1 option for training
        type=str,
        required=True,
        help="Path to and input CSV file containing SMILES, either for training or for making predictions. If training, also contains the associated target values.",
    )
    parser.add_argument(
        "--logdir",
        nargs="?",
        const="chemprop_logs",
        help="runs will be logged to {logdir}/chemprop_{time}.log. If unspecified, will use 'save_dir'. If only the flag is given (i.e., '--logdir'), then will write to 'chemprop_logs'",
    )

    data_args = parser.add_argument_group("input data parsing args")
    data_args.add_argument(
        "-s",
        "--smiles-columns",
        type=list,
        # to do: make sure default is coded correctly
        help="List of names of the columns containing SMILES strings. By default, uses the first :code:`number_of_molecules` columns.",
    )
    data_args.add_argument(
        "--number-of-molecules",
        type=int,
        default=1,
        help="Number of molecules in each input to the model. This must equal the length of :code:`smiles_columns` (if not :code:`None`).",
    )
    # to do: as we plug the three checkpoint options, see if we can reduce from three option to two or to just one.
    #        similar to how --features-path is/will be implemented
    data_args.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory from which to load model checkpoints (walks directory and ensembles all models that are found).",
    )
    data_args.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to model checkpoint (:code:`.pt` file).",
    )
    data_args.add_argument(
        "--checkpoint-paths",
        type=list[str],
        help="List of paths to model checkpoints (:code:`.pt` files).",
    )
    data_args.add_argument(
        "--no-cuda",
        action="store_true",
        help="Turn off cuda (i.e., use CPU instead of GPU).",
    )
    data_args.add_argument(
        "--gpu",
        type=int,
        help="Which GPU to use.",
    )
    data_args.add_argument(
        "--max-data-size",
        type=int,
        help="Maximum number of data points to load.",
    )
    data_args.add_argument(
        "-c",
        "--n-cpu",
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for the parallel data loading (0 means sequential).",
    )
    parser.add_argument("-g", "--n-gpu", type=int, default=1, help="the number of GPU(s) to use")
    data_args.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=50,
        help="Batch size.",
    )
    # to do: The next two arguments aren't in v1. See what they do in v2.
    data_args.add_argument(
        "--no-header-row", action="store_true", help="if there is no header in the input data CSV"
    )
    data_args.add_argument(
        "--rxn-idxs",
        nargs="+",
        type=int,
        default=list(),
        help="the indices in the input SMILES containing reactions. Unless specified, each input is assumed to be a molecule. Should be a number in `[0, N)`, where `N` is the number of `--smiles_columns` specified",
    )
    
    featurization_args = parser.add_argument_group("featurization args")
    featurization_args.add_argument(
        "--rxn-mode",
        "--reaction-mode",
        choices=RxnMode.keys(), 
        default="reac_diff",
        help="""
             Choices for construction of atom and bond features for reactions
             :code:`reac_prod`: concatenates the reactants feature with the products feature.
             :code:`reac_diff`: concatenates the reactants feature with the difference in features between reactants and products.
             :code:`prod_diff`: concatenates the products feature with the difference in features between reactants and products.
             :code:`reac_prod_balance`: concatenates the reactants feature with the products feature, balances imbalanced reactions.
             :code:`reac_diff_balance`: concatenates the reactants feature with the difference in features between reactants and products, balances imbalanced reactions.
             :code:`prod_diff_balance`: concatenates the products feature with the difference in features between reactants and products, balances imbalanced reactions.
             """,
    )
    featurization_args.add_argument(
        "--keep-h", 
        action="store_true",
        help="Whether H are explicitly specified in input (and should be kept this way). This option is intended to be used with the :code:`reaction` or :code:`reaction_solvent` options, and applies only to the reaction part.",
    )
    featurization_args.add_argument(
        "--adding-h", 
        action="store_true",
        help="Whether RDKit molecules will be constructed with adding the Hs to them. This option is intended to be used with Chemprop's default molecule or multi-molecule encoders, or in :code:`reaction_solvent` mode where it applies to the solvent only.",
    )
    featurization_args.add_argument(
        "--features-generators",
        action=RegistryAction(MoleculeFeaturizerRegistry),
        help="Method(s) of generating additional features.",
    )
    featurization_args.add_argument(
        "--features-path",
        type=list[str], # maybe should be type=str
        help="Path(s) to features to use in FNN (instead of features_generator).",
    )
    featurization_args.add_argument(
        "--phase_features_path",
        type=str,
        help="Path to features used to indicate the phase of the data in one-hot vector form. Used in spectra datatype.",
    )
    featurization_args.add_argument(
        "--no_features_scaling",
        action="store_true",
        help="Turn off scaling of features.",
    )
    featurization_args.add_argument(
        "--no_atom_descriptor_scaling",
        action="store_true",
        help="Turn off atom feature scaling.",
    )
    featurization_args.add_argument(
        "--no_bond_descriptor_scaling",
        action="store_true",
        help="Turn off bond feature scaling.",
    )
    featurization_args.add_argument(
        "--atom_features_path",
        type=str,
        help="Path to the extra atom features. Used as atom features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--atom_descriptors_path",
        type=str,
        help="Path to the extra atom descriptors. Used as descriptors and concatenated to the machine learned atomic representation.",
    )
    featurization_args.add_argument(
        "--overwrite_default_atom_features",
        action="store_true",
        help="Overwrites the default atom descriptors with the new ones instead of concatenating them. Can only be used if atom_descriptors are used as a feature.",
    )
    featurization_args.add_argument(
        "--bond_features_path",
        type=str,
        help="Path to the extra bond features. Used as bond features to featurize a given molecule.",
    )
    featurization_args.add_argument(
        "--bond_descriptors_path",
        type=str,
        help="Path to the extra bond descriptors. Used as descriptors and concatenated to the machine learned bond representation.",
    )
    featurization_args.add_argument(
        "--overwrite_default_bond_features",
        action="store_true",
        help="Overwrites the default bond descriptors with the new ones instead of concatenating them. Can only be used if bond_descriptors are used as a feature.",
    )
    # to do: remove these caching arguments after checking that the v2 code doesn't try to cache.
    # parser.add_argument(
    #     "--no_cache_mol",
    #     action="store_true",
    #     help="Whether to not cache the RDKit molecule for each SMILES string to reduce memory usage (cached by default).",
    # )
    # parser.add_argument(
    #     "--empty_cache",
    #     action="store_true",
    #     help="Whether to empty all caches before training or predicting. This is necessary if multiple jobs are run within a single script and the atom or bond features change.",
    # )
    # parser.add_argument(
    #     "--cache_cutoff",
    #     type=float,
    #     default=10000,
    #     help="Maximum number of molecules in dataset to allow caching. Below this number, caching is used and data loading is sequential. Above this number, caching is not used and data loading is parallel. Use 'inf' to always cache.",
    # )
    parser.add_argument(
        "--constraints-path",
        type=str,
        help="Path to constraints applied to atomic/bond properties prediction.",
    )

    # to do: see if we need to add functions from CommonArgs 
    return parser

def process_common_args(args: Namespace) -> Namespace:
    args.smiles_columns = (args.smiles_columns or list(range(args.number_of_molecules)))
    return args

def validate_common_args(args):
    pass

