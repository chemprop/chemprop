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

from chemprop.v2.cli.utils import CHK_DIR, column_str_to_int

from chemprop.v2.cli.common import add_common_args, process_common_args, validate_common_args

logger = logging.getLogger(__name__)


class TrainSubcommand(Subcommand):
    COMMAND = "train"
    HELP = "train a chemprop model"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        return add_train_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        validate_common_args(args)
        args = process_train_args(args)
        validate_train_args(args)
        main(args)

def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "-o",
        "--output-dir",
        "--save-dir",
        type=str,
        help="Directory where model checkpoints will be saved. Defaults to a directory in the current working directory with the same base name as the input file.",
    )
    # to do: see if we can tell lightning how often to log training loss
    parser.add_argument(
        "--log-frequency",
        type=int,
        default=10,
        help="The number of batches between each logging of the training loss.",
    )
    parser.add_argument(
        "--checkpoint-frzn",
        type=str,
        help="Path to model checkpoint file to be loaded for overwriting and freezing weights."
    )
    parser.add_argument(
        "--frzn-ffn-layers",
        type=int,
        default=0,
        help="Overwrites weights for the first n layers of the ffn from checkpoint model (specified checkpoint_frzn), where n is specified in the input. Automatically also freezes mpnn weights.",
    )
    parser.add_argument(
        "--freeze-first-only",
        action="store_true",
        help="Determines whether or not to use checkpoint_frzn for just the first encoder. Default (False) is to use the checkpoint to freeze all encoders. (only relevant for number_of_molecules > 1, where checkpoint model has number_of_molecules = 1)",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Whether to save test split predictions during training.",
    )
    parser.add_argument(
        "--resume-experiment",
        action="store_true",
        help="Whether to resume the experiment. Loads test results from any folds that have already been completed and skips training those folds.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to a :code:`.json` file containing arguments. Any arguments present in the config file will override arguments specified via the command line or by the defaults.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of models in ensemble.",
    )
    parser.add_argument( # to do: It looks like `reaction` is set later in main() based on rxn-idxs. Is that correct and do we need this argument?
        "--reaction",
        action="store_true",
        help="Whether to adjust MPNN layer to take reactions as input instead of molecules.", 
    )
    parser.add_argument(
        "--is-atom-bond-targets",
        action="store_true",
        help="Whether this is atomic/bond properties prediction.",
    )
    parser.add_argument(
        "--no-adding-bond-types",
        action="store_true",
        help="Whether the bond types determined by RDKit molecules added to the output of bond targets. This option is intended to be used with the :code:`is_atom_bond_targets`.",
    )
    parser.add_argument(
        "--keeping-atom-map",
        action="store_true",
        help="Whether RDKit molecules keep the original atom mapping. This option is intended to be used when providing atom-mapped SMILES with the :code:`is_atom_bond_targets`.",
    )

    mp_args = parser.add_argument_group("message passing")
    mp_args.add_argument(
        "--message-hidden-dim", type=int, default=300, help="hidden dimension of the messages"
    )
    mp_args.add_argument(
        "--bias", action="store_true", help="add bias to the message passing layers"
    )
    mp_args.add_argument(
        "--depth", type=int, default=3, help="Number of message passing steps."
    )
    mp_args.add_argument(
        "--undirected", action="store_true", help="Pass messages on undirected bonds/edges (always sum the two relevant bond vectors)."
    )
    mp_args.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="dropout probability in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--mpn-shared",
        action="store_true",
        help="Whether to use the same message passing neural network for all input molecules. Only relevant if :code:`number_of_molecules > 1`",
    )
    mp_args.add_argument(
        "--activation",
        default="relu",
        choices=['relu', 'leakyrelu', 'prelu', 'tanh', 'selu', 'elu'], # to do: should these be lowercase?
        help="activation function in message passing/FFN layers",
    )
    mp_args.add_argument(
        "--aggregation",
        "--agg",
        default="mean",
        action=RegistryAction(AggregationRegistry),
        help="the aggregation mode to use during graph readout",
    )
    mp_args.add_argument(
        "--aggregation-norm", type=float, default=100, help="normalization factor by which to divide summed up atomic features for 'norm' aggregation"
    )
    mp_args.add_argument(
        "--atom-messages", action="store_true", help="pass messages on atoms rather than bonds"
    )

    mpsolv_args = parser.add_argument_group("message passing with solvent")
    mpsolv_args.add_argument(
        "--reaction-solvent",
        action="store_true",
        help="Whether to adjust the MPNN layer to take as input a reaction and a molecule, and to encode them with separate MPNNs.",
    )
    mpsolv_args.add_argument(
        "--bias-solvent",
        action="store_true",
        help="Whether to add bias to linear layers for solvent MPN if :code:`reaction_solvent` is True.",
    )
    mpsolv_args.add_argument(
        "--hidden-size-solvent",
        type=int,
        default=300,
        help="Dimensionality of hidden layers in solvent MPN if :code:`reaction_solvent` is True.",
    )
    mpsolv_args.add_argument(
        "--depth-solvent",
        type=int,
        default=3,
        help="Number of message passing steps for solvent if :code:`reaction_solvent` is True.",
    )

    ffn_args = parser.add_argument_group("FFN args")
    ffn_args.add_argument( # to do: In v1 the mpn and fnn defaulted to the same hidden dim size. Now they can be different and have to be set separately. Do we want to change fnn_hidden_dims if message_hidden_dim is changed?
        "--ffn-hidden-dim", type=int, default=300, help="hidden dimension in the FFN top model"
    )
    ffn_args.add_argument( # to do: the default in v1 was 2. (see weights_ffn_num_layers option) Do we really want the default to now be 1?
        "--ffn-num-layers", type=int, default=1, help="number of layers in FFN top model"
    )
    ffn_args.add_argument(
        "--weights-ffn-num-layers",
        type=int,
        default=2,
        help="Number of layers in FFN for determining weights used in constrained targets.",
    )
    ffn_args.add_argument(
        "--features-only",
        action="store_true",
        help="Use only the additional features in an FFN, no graph network.",
    )
    ffn_args.add_argument(
        "--no-shared-atom-bond-ffn",
        action="store_true",
        help="Whether the FFN weights for atom and bond targets should be independent between tasks.",
    )

    exta_mpnn_args = parser.add_argument_group("extra MPNN args")
    exta_mpnn_args.add_argument(
        "--multiclass-num-classes",
        type=int,
        default=3,
        help="Number of classes when running multiclass classification.",
    )
    exta_mpnn_args.add_argument(
        "--spectral-activation", 
        default="exp", 
        choices=["softplus", "exp"], 
        help="Indicates which function to use in dataset_type spectra training to constrain outputs to be positive.",
    )

    data_args = parser.add_argument_group("input data parsing args")
    # data_args is added in add_common_args()
    data_args.add_argument(
        "--target-columns",
        type=list,
        help="Name of the columns containing target values. By default, uses all columns except the SMILES column and the :code:`ignore_columns`.",
    )
    data_args.add_argument(
        "--ignore-columns",
        type=list,
        help="Name of the columns to ignore when :code:`target_columns` is not provided.",
    )
   
    data_args.add_argument(
        "-t", 
        "--task", 
        "--dataset-type",
        default="regression", 
        action=RegistryAction(ReadoutRegistry), # to do: is this correct? The choices should be ['regression', 'classification', 'multiclass', 'spectra']
        help="Type of dataset. This determines the default loss function used during training.",
    )
    data_args.add_argument(
        "--spectra-phase-mask-path",
        type=str,
        help="Path to a file containing a phase mask array, used for excluding particular regions in spectra predictions."
    )
    data_args.add_argument(
        "--data-weights-path",
        type=str,
        help="a plaintext file that is parallel to the input data file and contains a single float per line that corresponds to the weight of the respective input weight during training. v1 help message: Path to weights for each molecule in the training data, affecting the relative weight of molecules in the loss function.",
    )
    data_args.add_argument(
        "--separate-val-path",
        type=str,
        help="Path to separate val set, optional.",
    )
    data_args.add_argument(
        "--separate-val-features-path",
        type=list[str],
        help="Path to file with features for separate val set.",
    )
    data_args.add_argument(
        "--separate-val-phase-features-path",
        type=str,
        help="Path to file with phase features for separate val set.",
    )
    data_args.add_argument(
        "--separate-val-atom-descriptors-path",
        type=str,
        help="Path to file with extra atom descriptors for separate val set.",
    )
    data_args.add_argument(
        "--separate-val-bond-descriptors-path",
        type=str,
        help="Path to file with extra atom descriptors for separate val set.",
    )
    data_args.add_argument(
        "--separate-val-constraints-path",
        type=str,
        help="Path to file with constraints for separate val set.",
    )
    data_args.add_argument("--val-atom-features-path") # to do: find what these were in v1 or if they were new in v2
    data_args.add_argument("--val-bond-features-path")

    data_args.add_argument(
        "--separate-test-path",
        type=str,
        help="Path to separate test set, optional.",
    )
    data_args.add_argument(
        "--separate-test-features-path",
        type=list[str],
        help="Path to file with features for separate test set.",
    )
    data_args.add_argument(
        "--separate-test-phase-features-path",
        type=str,
        help="Path to file with phase features for separate test set.",
    )
    data_args.add_argument(
        "--separate-test-atom-descriptors-path",
        type=str,
        help="Path to file with extra atom descriptors for separate test set.",
    )
    data_args.add_argument(
        "--separate-test-bond-descriptors-path",
        type=str,
        help="Path to file with extra atom descriptors for separate test set.",
    )
    data_args.add_argument(
        "--separate-test-constraints-path",
        type=str,
        help="Path to file with constraints for separate test set.",
    )
    data_args.add_argument("--test-atom-features-path") # to do: find what these were in v1 or if they were new in v2, it probably some combination of the arguments above.
    data_args.add_argument("--test-bond-features-path")

    train_args = parser.add_argument_group("training args")
    train_args.add_argument(
        "--target-weights",
        type=list[float],
        help="Weights associated with each target, affecting the relative weight of targets in the loss function. Must match the number of target columns.",
    )
    train_args.add_argument(
        "-l",
        "--loss-function",
        action=RegistryAction(LossFunctionRegistry),
    )
    train_args.add_argument(
        "--v-kl", 
        "--evidential-regularization",
        type=float, 
        default=0.2, # to do: the default in v1 was 0. Do we want it to default to 0.2 in v2?
        help="Value used in regularization for evidential loss function. The default value recommended by Soleimany et al.(2021) is 0.2. Optimal value is dataset-dependent; it is recommended that users test different values to find the best value for their model."
    )

    train_args.add_argument(
        "--eps", type=float, default=1e-8, help="evidential regularization epsilon"
    )

    train_args.add_argument( # to do: Is threshold the same thing as the spectra target floor? I'm not sure but combined them. 
        "-T", 
        "--threshold", 
        "--spectra-target-floor",
        type=float, 
        default=1e-8,
        help="spectral threshold limit. v1 help string: Values in targets for dataset type spectra are replaced with this value, intended to be a small positive number used to enforce positive values.")
    train_args.add_argument(
        "--metric"
        "--metrics",
        nargs="+",
        action=RegistryAction(MetricRegistry),
        help="evaluation metrics. If unspecified, will use the following metrics for given dataset types: regression->rmse, classification->roc, multiclass->ce ('cross entropy'), spectral->sid. If multiple metrics are provided, the 0th one will be used for early stopping and checkpointing",
    )
    train_args.add_argument(
        "--show-individual-scores",
        action="store_true",
        help="Show all scores for individual targets, not just average, at the end.",
    )
    train_args.add_argument( # to do: What is this for? I don't see it in v1.
        "-tw",
        "--task-weights",
        nargs="+",
        type=float,
        help="the weight to apply to an individual task in the overall loss",
    )
    train_args.add_argument(
        "--warmup-epochs", 
        type=int, # to do: This was a float in v1. I'm not sure why so I think int is better.
        default=2,
        help="Number of epochs during which learning rate increases linearly from :code:`init_lr` to :code:`max_lr`. Afterwards, learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr`.",
    )

    train_args.add_argument("--num-lrs", type=int, default=1)
    
    train_args.add_argument(
        "--init-lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    train_args.add_argument(
        "--max-lr", 
        type=float, 
        default=1e-3,
        help="Maximum learning rate.",
    )
    train_args.add_argument(
        "--final-lr", 
        type=float, 
        default=1e-4,
        help="Final learning rate.",
    )
    train_args.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="the number of epochs to train over"
    )
    train_args.add_argument(
        "--grad-clip",
        type=float,
        help="Maximum magnitude of gradient during training.",
    )
    train_args.add_argument(
        "--class-balance",
        action="store_true",
        help="Trains with an equal number of positives and negatives in each batch.",
    )

    split_args = parser.add_argument_group("split args")
    split_args.add_argument(
        "--split",
        "--split-type",
        default="random",
        choices=['random', 'scaffold'],#'scaffold_balanced', 'predetermined', 'crossval', 'cv', 'cv-no-test', 'index_predetermined', 'random_with_repeated_smiles'],
        help="Method of splitting the data into train/val/test.",
    )
    split_args.add_argument(
        "--split-sizes",
        type=list[float],
        default=[0.8, 0.1, 0.1],
        help="Split proportions for train/validation/test sets.",    
    )
    # split_args.add_argument(
    #     "--split-key-molecule",
    #     type=int,
    #     default=0,
    #     help="The index of the key molecule used for splitting when multiple molecules are present and constrained split_type is used, like scaffold_balanced or random_with_repeated_smiles.       Note that this index begins with zero for the first molecule.",
    # )
    split_args.add_argument(
        "-k", 
        "--num-folds", 
        type=int, 
        default=1,
        help="Number of folds when performing cross validation.",
    )
    split_args.add_argument(
        "--folds-file",
        type=str,
        help="Optional file of fold labels.",
    )
    split_args.add_argument(
        "--val-fold-index",
        type=int,
        help="Which fold to use as val for leave-one-out cross val.",
    )
    split_args.add_argument(
        "--test-fold-index",
        type=int,
        help="Which fold to use as test for leave-one-out cross val.",
    )
    split_args.add_argument(
        "--crossval-index-dir",
        type=str,
        help="Directory in which to find cross validation index files.",
    )
    split_args.add_argument(
        "--crossval-index-file",
        type=str,
        help="Indices of files to use as train/val/test. Overrides :code:`--num_folds` and :code:`--seed`.",
    )
    split_args.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to use when splitting data into train/val/test sets. When :code`num_folds > 1`, the first fold uses this seed and all subsequent folds add 1 to the seed.",
    )
    split_args.add_argument(
        "--save-smiles-splits",
        action="store_true",
        help="Save smiles for each train/val/test splits for prediction convenience later.",
    )

    parser.add_argument( # to do: do we need this?
        "--pytorch-seed",
        type=int,
        default=0,
        help="Seed for PyTorch randomness (e.g., random initial weights).",
    )

    return parser

def process_train_args(args: Namespace) -> Namespace:
    args.output_dir = Path(args.output_dir or Path.cwd() / args.input.stem)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    args.ignore_columns = column_str_to_int(args.ignore_columns, args.header)
    args.target_columns = column_str_to_int(args.target_columns, args.header)

    if args.target_columns is None:
        ignore_columns = set(args.smiles_columns + ([] if args.ignore_columns is None else args.ignore_columns))
        args.target_columns = [i for i in range(len(args.header)) if i not in ignore_columns]

    return args

def validate_train_args(args):
    pass


def main(args):
    bond_messages = not args.atom_messages
    n_tasks = len(args.target_columns)
    bounded = args.loss_function is not None and "bounded" in args.loss_function

    if args.number_of_molecules > 1:
        warnings.warn(
            "Multicomponent input is not supported at this time! Using only the 1st input..."
        )

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_columns=args.smiles_columns,
        target_columns=args.target_columns,
        bounded=bounded,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators,
        keep_h=args.keep_h,
        add_h=args.add_h,
        reaction=0 in args.rxn_idxs, # to do: check if this is correct
    )

    all_data = build_data_from_files(
        args.input,
        **format_kwargs,
        p_features=args.features_path,
        p_atom_feats=args.atom_features_path,
        p_bond_feats=args.bond_features_path,
        p_atom_descs=args.atom_descriptors_path,
        **featurization_kwargs,
    )

    if args.val_path is None and args.test_path is None:
        train_data, val_data, test_data = split_data(all_data, args.split, args.split_sizes)
    elif args.test_path is not None:
        test_data = build_data_from_files(
            args.test_path,
            p_features=args.test_features_path,
            p_atom_feats=args.test_atom_features_path,
            p_bond_feats=args.test_bond_features_path,
            p_atom_descs=args.test_atom_descriptors_path,
            **format_kwargs,
            **featurization_kwargs,
        )
        if args.val_path is not None:
            val_data = build_data_from_files(
                args.val_path,
                p_features=args.val_features_path,
                p_atom_feats=args.val_atom_features_path,
                p_bond_feats=args.val_bond_features_path,
                p_atom_descs=args.val_atom_descriptors_path,
                **format_kwargs,
                **featurization_kwargs,
            )
            train_data = all_data
        else:
            train_data, val_data, _ = split_data(all_data, args.split, args.split_sizes)
    else:
        raise ArgumentError("'val_path' must be specified is 'test_path' is provided!")
    logger.info(f"train/val/test sizes: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    train_dset = make_dataset(train_data, bond_messages, args.rxn_mode)
    val_dset = make_dataset(val_data, bond_messages, args.rxn_mode)

    mp_cls = BondMessageBlock if bond_messages else AtomMessageBlock
    mp_block = mp_cls(
        train_dset.featurizer.atom_fdim,
        train_dset.featurizer.bond_fdim,
        args.message_hidden_dim,
        args.message_bias,
        args.depth,
        args.undirected,
        args.dropout,
        args.activation,
    )
    agg = Factory.build(AggregationRegistry[args.aggregation], norm=args.norm)
    readout_cls = ReadoutRegistry[args.readout]

    if args.loss_function is not None:
        criterion = Factory.build(
            LossFunctionRegistry[args.loss_function],
            v_kl=args.v_kl,
            threshold=args.threshold,
            eps=args.eps,
        )
    else:
        logger.info(
            f"No loss function specified, will use class default: {readout_cls._default_criterion}"
        )
        criterion = readout_cls._default_criterion

    readout_ffn = Factory.build(
        readout_cls,
        input_dim=mp_block.output_dim,
        n_tasks=args.n_tasks,
        hidden_dim=args.ffn_hidden_dim,
        n_layers=args.ffn_num_layers,
        dropout=args.dropout,
        activation=args.activation,
        criterion=criterion,
        num_classes=args.num_classes,
        spectral_activation=args.spectral_activation,
    )

    if isinstance(readout_ffn, RegressionFFN):
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)
        logger.info(f"Train data: loc = {scaler.mean_}, scale = {scaler.scale_}")
    else:
        scaler = None

    train_loader = data.MolGraphDataLoader(train_dset, args.batch_size, args.n_cpu)
    val_loader = data.MolGraphDataLoader(val_dset, args.batch_size, args.n_cpu, shuffle=False)
    if len(test_data) > 0:
        test_dset = make_dataset(test_data, bond_messages, args.rxn_mode)
        test_loader = data.MolGraphDataLoader(test_dset, args.batch_size, args.n_cpu, shuffle=False)
    else:
        test_loader = None

    model = MPNN(
        mp_block,
        agg,
        readout_ffn,
        None,
        args.task_weights,
        args.warmup_epochs,
        args.num_lrs,
        args.init_lr,
        args.max_lr,
        args.final_lr,
    )
    logger.info(model)

    monitor_mode = "min" if model.metrics[0].minimize else "max"
    logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

    tb_logger = TensorBoardLogger(args.output_dir, "tb_logs")
    checkpointing = ModelCheckpoint(
        args.output_dir / "chkpts",
        "{epoch}-{val_loss:.2f}",
        "val_loss",
        mode=monitor_mode,
        save_last=True,
    )
    early_stopping = EarlyStopping("val_loss", patience=5, mode=monitor_mode)

    trainer = pl.Trainer(
        logger=tb_logger,
        enable_progress_bar=True,
        accelerator="auto",
        devices=args.n_gpu if torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
        callbacks=[checkpointing, early_stopping],
    )
    trainer.fit(model, train_loader, val_loader)

    if test_loader is not None:
        if args.dataset_type == "regression":
            model.loc, model.scale = float(scaler.mean_), float(scaler.scale_)
        results = trainer.test(model, test_loader)[0]
        logger.info(f"Test results: {results}")

    p_model = args.output / "model.pt"
    torch.save(model.state_dict(), p_model)
    logger.info(f"model state dict saved to '{p_model}'")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    process_args(args)

    main(args)
