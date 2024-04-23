import json
import logging
import sys
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from chemprop.cli.common import add_common_args, process_common_args, validate_common_args
from chemprop.cli.train import (
    add_train_args,
    build_datasets,
    build_model,
    build_splits,
    normalize_inputs,
    process_train_args,
    validate_train_args,
)
from chemprop.cli.utils.command import Subcommand
from chemprop.data import build_dataloader
from chemprop.nn import AggregationRegistry
from chemprop.nn.utils import Activation
from chemprop.nn.transforms import UnscaleTransform

NO_RAY = False
DEFAULT_SEARCH_SPACE = {}
try:
    from ray import tune
    from ray.train import CheckpointConfig, RunConfig, ScalingConfig
    from ray.train.lightning import (
        RayDDPStrategy,
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    from ray.train.torch import TorchTrainer
    from ray.tune.schedulers import ASHAScheduler

    DEFAULT_SEARCH_SPACE = {
        "activation": tune.choice(categories=list(Activation.keys())),
        "aggregation": tune.choice(categories=list(AggregationRegistry.keys())),
        "aggregation_norm": tune.quniform(lower=1, upper=200, q=1),
        "batch_size": tune.loguniform(lower=16, upper=256, base=2),
        "depth": tune.quniform(lower=2, upper=6, q=1),
        "dropout": tune.choice([tune.choice([0.0]), tune.quniform(lower=0.05, upper=0.4, q=0.05)]),
        "ffn_hidden_dim": tune.quniform(lower=300, upper=2400, q=100),
        "ffn_num_layers": tune.quniform(lower=1, upper=3, q=1),
        "final_lr_ratio": tune.loguniform(lower=1e-4, upper=1),
        "message_hidden_dim": tune.quniform(lower=300, upper=2400, q=100),
        "init_lr_ratio": tune.loguniform(lower=1e-4, upper=1),
        "max_lr": tune.loguniform(lower=1e-6, upper=1e-2),
        "warmup_epochs": None,
    }
except ImportError:
    NO_RAY = True

NO_HYPEROPT = False
try:
    from ray.tune.search.hyperopt import HyperOptSearch
except ImportError:
    NO_HYPEROPT = True

# NO_OPTUNA = False
# try:
#     from ray.tune.search.optuna import OptunaSearch
# except ImportError:
#     NO_OPTUNA = True


logger = logging.getLogger(__name__)

SEARCH_SPACE = DEFAULT_SEARCH_SPACE

SEARCH_PARAM_KEYWORDS_MAP = {
    "basic": ["depth", "ffn_num_layers", "dropout", "ffn_hidden_dim", "message_hidden_dim"],
    "learning_rate": ["max_lr", "init_lr_ratio", "final_lr_ratio", "warmup_epochs"],
    "all": list(DEFAULT_SEARCH_SPACE.keys()),
}


class HpoptSubcommand(Subcommand):
    COMMAND = "hpopt"
    HELP = "perform hyperparameter optimization on the given task"

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = add_common_args(parser)
        parser = add_train_args(parser)
        return add_hpopt_args(parser)

    @classmethod
    def func(cls, args: Namespace):
        args = process_common_args(args)
        args = process_train_args(args)
        args = process_hpopt_args(args)
        validate_common_args(args)
        validate_train_args(args)
        main(args)


def add_hpopt_args(parser: ArgumentParser) -> ArgumentParser:
    hpopt_args = parser.add_argument_group("Chemprop hyperparameter optimization arguments")

    hpopt_args.add_argument(
        "--search-parameter-keywords",
        type=str,
        nargs="+",
        default=["basic"],
        help=f"""The model parameters over which to search for an optimal hyperparameter configuration.
    Some options are bundles of parameters or otherwise special parameter operations.

    Special keywords:
        basic - the default set of hyperparameters for search: depth, ffn_num_layers, dropout, message_hidden_dim, and ffn_hidden_dim.
        learning_rate - search for max_lr, init_lr_ratio, final_lr_ratio, and warmup_epochs. The search for init_lr and final_lr values
            are defined as fractions of the max_lr value. The search for warmup_epochs is as a fraction of the total epochs used.
        all - include search for all 13 inidividual keyword options

    Individual supported parameters:
        {list(DEFAULT_SEARCH_SPACE.keys())}
    """,
    )

    hpopt_args.add_argument(
        "--hpopt-save-dir",
        type=Path,
        help="Directory to save the hyperparameter optimization results",
    )

    raytune_args = parser.add_argument_group("Ray Tune arguments")

    raytune_args.add_argument(
        "--raytune-num-samples",
        type=int,
        default=10,
        help="Passed directly to Ray Tune TuneConfig to control number of trials to run",
    )

    raytune_args.add_argument(
        "--raytune-search-algorithm",
        choices=["random", "hyperopt"],  # , "optuna"],
        default="hyperopt",
        help="Passed to Ray Tune TuneConfig to control search algorithm",
    )

    raytune_args.add_argument(
        "--raytune-num-workers",
        type=int,
        default=1,
        help="Passed directly to Ray Tune ScalingConfig to control number of workers to use",
    )

    raytune_args.add_argument(
        "--raytune-use-gpu",
        action="store_true",
        help="Passed directly to Ray Tune ScalingConfig to control whether to use GPUs",
    )

    raytune_args.add_argument(
        "--raytune-num-checkpoints-to-keep",
        type=int,
        default=1,
        help="Passed directly to Ray Tune CheckpointConfig to control number of checkpoints to keep",
    )

    raytune_args.add_argument(
        "--raytune-grace-period",
        type=int,
        default=10,
        help="Passed directly to Ray Tune ASHAScheduler to control grace period",
    )

    raytune_args.add_argument(
        "--raytune-reduction-factor",
        type=int,
        default=2,
        help="Passed directly to Ray Tune ASHAScheduler to control reduction factor",
    )

    hyperopt_args = parser.add_argument_group("Hyperopt arguments")

    hyperopt_args.add_argument(
        "--hyperopt-n-initial-points",
        type=int,
        default=20,
        help="Passed directly to HyperOptSearch to control number of initial points to sample",
    )

    hyperopt_args.add_argument(
        "--hyperopt-random-state-seed",
        type=int,
        default=None,
        help="Passed directly to HyperOptSearch to control random state seed",
    )

    return parser


def process_hpopt_args(args: Namespace) -> Namespace:
    if args.hpopt_save_dir is None:
        args.hpopt_save_dir = Path(f"chemprop_hpopt/{args.data_path.stem}")

    args.hpopt_save_dir.mkdir(exist_ok=True, parents=True)

    search_parameters = set()

    for keyword in args.search_parameter_keywords:
        if keyword not in SEARCH_PARAM_KEYWORDS_MAP and keyword not in SEARCH_SPACE:
            raise ValueError(
                f"Search parameter keyword: {keyword} not in available options: {list(SEARCH_PARAM_KEYWORDS_MAP.keys()) + list(SEARCH_SPACE.keys())}."
            )

        search_parameters.update(
            SEARCH_PARAM_KEYWORDS_MAP[keyword]
            if keyword in SEARCH_PARAM_KEYWORDS_MAP
            else [keyword]
        )

    args.search_parameter_keywords = list(search_parameters)

    return args


def build_search_space(search_parameters: list[str], train_epochs: int) -> dict:
    if "warmup_epochs" not in SEARCH_SPACE and "warmup_epochs" in search_parameters:
        SEARCH_SPACE["warmup_epochs"] = tune.quniform(lower=1, upper=train_epochs // 2, q=1)

    return {param: SEARCH_SPACE[param] for param in search_parameters}


def update_args_with_config(args: Namespace, config: dict) -> Namespace:
    args = deepcopy(args)

    for key, value in config.items():
        match key:
            case "final_lr_ratio":
                setattr(args, "final_lr", value * args.max_lr)

            case "init_lr_ratio":
                setattr(args, "init_lr", value * args.max_lr)

            case _:
                assert key in args, f"Key: {key} not found in args."
                setattr(args, key, value)

    return args


def train_model(config, args, train_dset, val_dset, logger, output_transform, input_transforms):
    update_args_with_config(args, config)

    train_loader = build_dataloader(
        train_dset, args.batch_size, args.num_workers, seed=args.data_seed
    )
    val_loader = build_dataloader(val_dset, args.batch_size, args.num_workers, shuffle=False)

    seed = args.pytorch_seed if args.pytorch_seed is not None else torch.seed()

    torch.manual_seed(seed)

    model = build_model(args, train_loader.dataset, output_transform, input_transforms)
    logger.info(model)

    monitor_mode = "min" if model.metrics[0].minimize else "max"
    logger.debug(f"Evaluation metric: '{model.metrics[0].alias}', mode: '{monitor_mode}'")

    patience = args.patience if args.patience is not None else args.epochs
    early_stopping = EarlyStopping("val_loss", patience=patience, mode=monitor_mode)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        gradient_clip_val=args.grad_clip,
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback(), early_stopping],
        plugins=[RayLightningEnvironment()],
        deterministic=args.pytorch_seed is not None,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, train_loader, val_loader)


def tune_model(
    args, train_dset, val_dset, logger, monitor_mode, output_transform, input_transforms
):
    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=min(args.raytune_grace_period, args.epochs),
        reduction_factor=args.raytune_reduction_factor,
    )

    scaling_config = ScalingConfig(
        num_workers=args.raytune_num_workers, use_gpu=args.raytune_use_gpu
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=args.raytune_num_checkpoints_to_keep,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order=monitor_mode,
    )

    run_config = RunConfig(
        checkpoint_config=checkpoint_config,
        storage_path=args.hpopt_save_dir.absolute() / "ray_results",
    )

    ray_trainer = TorchTrainer(
        lambda config: train_model(
            config, args, train_dset, val_dset, logger, output_transform, input_transforms
        ),
        scaling_config=scaling_config,
        run_config=run_config,
    )

    match args.raytune_search_algorithm:
        case "random":
            search_alg = None
        case "hyperopt":
            if NO_HYPEROPT:
                raise ImportError(
                    "HyperOptSearch requires hyperopt to be installed. Use 'pip -U install hyperopt' to install."
                )

            search_alg = HyperOptSearch(
                n_initial_points=args.hyperopt_n_initial_points,
                random_state_seed=args.hyperopt_random_state_seed,
            )
        # case "optuna":
        #     if NO_OPTUNA:
        #         raise ImportError(
        #             "OptunaSearch requires optuna to be installed. Use 'pip -U install optuna' to install."
        #         )

        #     search_alg = OptunaSearch()

    tune_config = tune.TuneConfig(
        metric="val_loss",
        mode=monitor_mode,
        num_samples=args.raytune_num_samples,
        scheduler=scheduler,
        search_alg=search_alg,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={
            "train_loop_config": build_search_space(args.search_parameter_keywords, args.epochs)
        },
        tune_config=tune_config,
    )

    return tuner.fit()


def main(args: Namespace):
    if NO_RAY:
        raise ImportError(
            "Ray Tune requires ray to be installed. Use 'pip -U install ray[tune]' to install."
        )

    format_kwargs = dict(
        no_header_row=args.no_header_row,
        smiles_cols=args.smiles_columns,
        rxn_cols=args.reaction_columns,
        target_cols=args.target_columns,
        ignore_cols=args.ignore_columns,
        splits_col=args.splits_column,
        weight_col=args.weight_column,
        bounded=args.loss_function is not None and "bounded" in args.loss_function,
    )
    featurization_kwargs = dict(
        features_generators=args.features_generators, keep_h=args.keep_h, add_h=args.add_h
    )

    train_data, val_data, test_data = build_splits(args, format_kwargs, featurization_kwargs)
    train_dset, val_dset, test_dset = build_datasets(args, train_data[0], val_data[0], test_data[0])

    input_transforms = normalize_inputs(train_dset, val_dset, args)

    if "regression" in args.task_type:
        output_scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(output_scaler)
        logger.info(f"Train data: mean = {output_scaler.mean_} | std = {output_scaler.scale_}")
        output_transform = UnscaleTransform.from_standard_scaler(output_scaler)
    else:
        output_transform = None

    train_loader = build_dataloader(
        train_dset, args.batch_size, args.num_workers, seed=args.data_seed
    )

    model = build_model(args, train_loader.dataset, output_transform, input_transforms)
    monitor_mode = "min" if model.metrics[0].minimize else "max"

    results = tune_model(
        args, train_dset, val_dset, logger, monitor_mode, output_transform, input_transforms
    )

    best_result = results.get_best_result()
    best_config = best_result.config
    best_checkpoint = best_result.checkpoint  # Get best trial's best checkpoint

    logger.info(f"Saving best hyperparameter parameters: {best_config}")

    with open(args.hpopt_save_dir / "best_params.json", "w") as f:
        json.dump(best_config, f, indent=4)

    logger.info(f"Saving best hyperparameter configuration checkpoint: {best_checkpoint}")

    torch.save(best_checkpoint, args.hpopt_save_dir / "best_checkpoint.ckpt")

    result_df = results.get_dataframe()

    logger.info(f"Saving hyperparameter optimization results: {result_df}")

    result_df.to_csv(args.hpopt_save_dir / "all_progress.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = HpoptSubcommand.add_args(parser)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
    args = parser.parse_args()
    HpoptSubcommand.func(args)
