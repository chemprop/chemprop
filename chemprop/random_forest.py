from argparse import Namespace
from logging import Logger
from pprint import pformat
from typing import Callable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange, tqdm

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.features import get_features_generator
from chemprop.train.evaluate import evaluate_predictions
from chemprop.utils import get_metric_func


def single_task_random_forest(train_data: MoleculeDataset,
                              test_data: MoleculeDataset,
                              metric_func: Callable,
                              args: Namespace,
                              logger: Logger = None) -> List[float]:
    scores = []
    num_tasks = train_data.num_tasks()
    for task_num in trange(num_tasks):
        # Only get features and targets for molecules where target is not None
        train_features, train_targets = zip(*[(features, targets[task_num])
                                              for features, targets in zip(train_data.features(), train_data.targets())
                                              if targets[task_num] is not None])
        test_features, test_targets = zip(*[(features, targets[task_num])
                                            for features, targets in zip(test_data.features(), test_data.targets())
                                            if targets[task_num] is not None])

        if args.dataset_type == 'regression':
            model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1)
        elif args.dataset_type == 'classification':
            model = RandomForestClassifier(class_weight=args.class_weight, n_estimators=args.num_trees, n_jobs=-1)
        else:
            raise ValueError(f'dataset_type "{args.dataset_type}" not supported.')

        model.fit(train_features, train_targets)

        test_preds = model.predict(test_features)

        test_preds = [[pred] for pred in test_preds]
        test_targets = [[target] for target in test_targets]

        score = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=1,
            metric_func=metric_func,
            dataset_type=args.dataset_type,
            logger=logger
        )
        scores.append(score[0])

    return scores


def multi_task_random_forest(train_data: MoleculeDataset,
                             test_data: MoleculeDataset,
                             metric_func: Callable,
                             args: Namespace,
                             logger: Logger = None) -> List[float]:
    num_tasks = train_data.num_tasks()

    if args.dataset_type == 'regression':
        model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1)
    elif args.dataset_type == 'classification':
        model = RandomForestClassifier(n_estimators=args.num_trees, n_jobs=-1)
    else:
        raise ValueError(f'dataset_type "{args.dataset_type}" not supported.')

    train_targets = train_data.targets()
    if train_data.num_tasks() == 1:
        train_targets = [targets[0] for targets in train_targets]

    model.fit(train_data.features(), train_targets)

    test_preds = model.predict(test_data.features())
    if num_tasks == 1:
        test_preds = [[pred] for pred in test_preds]

    scores = evaluate_predictions(
        preds=test_preds,
        targets=test_data.targets(),
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    return scores


def run_random_forest(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    metric_func = get_metric_func(args.metric)

    debug('Loading data')
    data = get_data(path=args.data_path)

    debug(f'Splitting data with seed {args.seed}')
    # Need to have val set so that train and test sets are the same as when doing MPN
    train_data, _, test_data = split_data(data=data, split_type=args.split_type, seed=args.seed, args=args)

    debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | test size = {len(test_data):,}')

    debug('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for dataset in [train_data, test_data]:
        for datapoint in tqdm(dataset, total=len(dataset)):
            datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    debug('Training')
    if args.single_task:
        scores = single_task_random_forest(train_data, test_data, metric_func, args, logger)
    else:
        scores = multi_task_random_forest(train_data, test_data, metric_func, args, logger)

    info(f'Test {args.metric} = {np.nanmean(scores)}')

    return scores


def cross_validate_random_forest(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print
    init_seed = args.seed

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        model_scores = run_random_forest(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

    # Report scores across folds
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    return mean_score, std_score
