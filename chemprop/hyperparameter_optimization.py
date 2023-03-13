"""Optimizes hyperparameters using Bayesian optimization."""

from copy import deepcopy
from typing import Dict, Union
import os
from functools import partial

from hyperopt import fmin, tpe, Trials
import numpy as np

from chemprop.args import HyperoptArgs
from chemprop.constants import HYPEROPT_LOGGER_NAME
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.train import cross_validate, run_training
from chemprop.utils import create_logger, makedirs, timeit
from chemprop.hyperopt_utils import merge_trials, load_trials, save_trials, \
    get_hyperopt_seed, load_manual_trials, build_search_space, save_config


@timeit(logger_name=HYPEROPT_LOGGER_NAME)
def hyperopt(args: HyperoptArgs) -> None:
    """
    Runs hyperparameter optimization on a Chemprop model.

    Hyperparameter optimization optimizes the following parameters:

    * :code:`hidden_size`: The hidden size of the neural network layers is selected from {300, 400, ..., 2400}
    * :code:`depth`: The number of message passing iterations is selected from {2, 3, 4, 5, 6}
    * :code:`dropout`: The dropout probability is selected from {0.0, 0.05, ..., 0.4}
    * :code:`ffn_num_layers`: The number of feed-forward layers after message passing is selected from {1, 2, 3}

    The best set of hyperparameters is saved as a JSON file to :code:`args.config_save_path`.

    :param args: A :class:`~chemprop.args.HyperoptArgs` object containing arguments for hyperparameter
                 optimization in addition to all arguments needed for training.
    """
    # Create logger
    logger = create_logger(name=HYPEROPT_LOGGER_NAME, save_dir=args.log_dir, quiet=True)

    # Build search space
    logger.info(f"Creating search space using parameters {args.search_parameters}.")
    space = build_search_space(
        search_parameters=args.search_parameters, train_epochs=args.epochs
    )
    int_keys = [
        "batch_size",
        "depth",
        "ffn_hidden_size",
        "ffn_num_layers",
        "hidden_size",
        "linked_hidden_size",
        "warmup_epochs",
    ]

    # Load in manual trials
    if args.manual_trial_dirs is not None:
        manual_trials = load_manual_trials(
            manual_trials_dirs=args.manual_trial_dirs,
            param_keys=space.keys(),
            hyperopt_args=args,
        )
        logger.info(
            f"{len(manual_trials)} manual trials included in hyperparameter search."
        )
    else:
        manual_trials = None
        logger.info("No manual trials loaded as part of hyperparameter search")

    makedirs(args.hyperopt_checkpoint_dir)

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]], seed: int) -> Dict:
        # Convert hyperparams from float to int when necessary
        for key in int_keys:
            if key in hyperparams:
                hyperparams[key] = int(hyperparams[key])

        # Copy args
        hyper_args = deepcopy(args)

        # Update args with hyperparams
        if args.save_dir is not None:
            folder_name = f"trial_seed_{seed}"
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)

        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        if "linked_hidden_size" in hyperparams:
            hyper_args.ffn_hidden_size = hyperparams["linked_hidden_size"]
            hyper_args.hidden_size = hyperparams["linked_hidden_size"]

        if "init_lr_ratio" in hyperparams:
            hyper_args.init_lr = hyperparams["max_lr"] * hyperparams["init_lr_ratio"]
        if "final_lr_ratio" in hyperparams:
            hyper_args.final_lr = hyperparams["max_lr"] * hyperparams["final_lr_ratio"]

        # Cross validate
        mean_score, std_score = cross_validate(args=hyper_args, train_func=run_training)

        # Record results
        temp_model = MoleculeModel(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f"Trial results with seed {seed}")
        logger.info(hyperparams)
        logger.info(f"num params: {num_params:,}")
        logger.info(f"{mean_score} +/- {std_score} {hyper_args.metric}")

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == "classification":
                mean_score = 0
            else:
                raise ValueError(
                    "Can't handle nan score for non-classification dataset."
                )

        loss = (1 if hyper_args.minimize_score else -1) * mean_score

        return {
            "loss": loss,
            "status": "ok",
            "mean_score": mean_score,
            "std_score": std_score,
            "hyperparams": hyperparams,
            "num_params": num_params,
            "seed": seed,
        }

    # Iterate over a number of trials
    for i in range(args.num_iters):
        # run fmin and load trials in single steps to allow for parallel operation
        trials = load_trials(
            dir_path=args.hyperopt_checkpoint_dir, previous_trials=manual_trials
        )
        if len(trials) > 0 and set(space.keys()) != set(trials.vals.keys()):
            raise ValueError(
                f"Loaded hyperopt checkpoints files must be searching over the same parameters as \
                    the hyperparameter optimization job. Loaded trials covered variation in the parameters {set(trials.vals.keys())}. \
                    The current search is over the parameters {set(space.keys())}."
            )
        if len(trials) >= args.num_iters:
            break

        # Set a unique random seed for each trial. Pass it into objective function for logging purposes.
        hyperopt_seed = get_hyperopt_seed(
            seed=args.hyperopt_seed, dir_path=args.hyperopt_checkpoint_dir
        )
        fmin_objective = partial(objective, seed=hyperopt_seed)
        os.environ["HYPEROPT_FMIN_SEED"] = str(
            hyperopt_seed
        )  # this environment variable changes the seed in fmin

        # Log the start of the trial
        logger.info(f"Initiating trial with seed {hyperopt_seed}")
        logger.info(f"Loaded {len(trials)} previous trials")
        if len(trials) < args.startup_random_iters:
            random_remaining = args.startup_random_iters - len(trials)
            logger.info(
                f"Parameters assigned with random search, {random_remaining} random trials remaining"
            )
        else:
            logger.info(f"Parameters assigned with TPE directed search")

        fmin(
            fmin_objective,
            space,
            algo=partial(tpe.suggest, n_startup_jobs=args.startup_random_iters),
            max_evals=len(trials) + 1,
            trials=trials,
        )

        # Create a trials object with only the last instance by merging the last data with an empty trials object
        last_trial = merge_trials(Trials(), [trials.trials[-1]])
        save_trials(args.hyperopt_checkpoint_dir, last_trial, hyperopt_seed, logger)

    # Report best result
    all_trials = load_trials(
        dir_path=args.hyperopt_checkpoint_dir, previous_trials=manual_trials
    )
    results = all_trials.results
    results = [result for result in results if not np.isnan(result["mean_score"])]
    best_result = min(
        results,
        key=lambda result: (1 if args.minimize_score else -1) * result["mean_score"],
    )
    logger.info(f'Best trial, with seed {best_result["seed"]}')
    logger.info(best_result["hyperparams"])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(
        f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}'
    )

    # Save best hyperparameter settings as JSON config file
    save_config(
        config_path=args.config_save_path,
        hyperparams_dict=best_result["hyperparams"],
        max_lr=args.max_lr,
    )


def chemprop_hyperopt() -> None:
    """Runs hyperparameter optimization for a Chemprop model.

    This is the entry point for the command line command :code:`chemprop_hyperopt`.
    """
    hyperopt(args=HyperoptArgs().parse_args())
