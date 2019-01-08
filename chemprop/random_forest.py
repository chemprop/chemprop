from argparse import Namespace
from logging import Logger
from pprint import pformat
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from chemprop.data.utils import get_data, split_data
from chemprop.features import morgan_fingerprint
from chemprop.train.evaluate import evaluate_predictions
from chemprop.utils import get_metric_func


def run_random_forest(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    metric_func = get_metric_func(args.metric)

    debug('Loading data')
    data = get_data(args.data_path)

    debug('Splitting data with seed {}'.format(args.seed))
    train_data, val_data, test_data = split_data(data=data, split_type=args.split_type)

    debug('Total size = {:,} | train size = {:,} | val size = {:,} | test size = {:,}'.format(
        len(data), len(train_data), len(val_data), len(test_data)))

    debug('Computing morgan fingerprints')
    for dataset in [train_data, val_data, test_data]:
        for datapoint in tqdm(dataset, total=len(dataset)):
            datapoint.set_features(morgan_fingerprint(smiles=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    debug('Building model')
    if args.dataset_type == 'regression':
        model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1)
    elif args.dataset_type == 'classification':
        model = RandomForestClassifier(n_estimators=args.num_trees, n_jobs=-1)
    else:
        raise ValueError('dataset_type "{}" not supported.'.format(args.dataset_type))

    debug('Training model')
    train_targets = train_data.targets()
    if train_data.num_tasks() == 1:
        train_targets = [targets[0] for targets in train_targets]

    model.fit(train_data.features(), train_targets)

    debug('Predicting')
    test_preds = model.predict(test_data.features())
    if train_data.num_tasks() == 1:
        test_preds = [[pred] for pred in test_preds]

    debug('Evaluating')
    score = evaluate_predictions(
        preds=test_preds,
        targets=test_data.targets(),
        metric_func=metric_func,
        dataset_type=args.dataset_type
    )
    info('Test {} = {}'.format(args.metric, np.nanmean(score)))

    return score


def cross_validate_random_forest(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print
    init_seed = args.seed

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        model_scores = run_random_forest(args, logger)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info('Seed {} ==> test {} = {:.6f}'.format(init_seed + fold_num, args.metric, np.nanmean(scores)))

    # Report scores across folds
    avg_scores = np.nanmean(all_scores, axis=1)  # average score for each model across tasks
    mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
    info('Overall test {} = {:.6f} +/- {:.6f}'.format(args.metric, mean_score, std_score))

    return mean_score, std_score
