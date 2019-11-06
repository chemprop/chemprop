from argparse import Namespace
from logging import Logger
import os
import pickle
from pprint import pformat
from typing import Callable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from tqdm import trange, tqdm

from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, split_data
from chemprop.features import get_features_generator
from chemprop.train.evaluate import evaluate_predictions
from chemprop.utils import get_metric_func, makedirs


def predict(model,
            model_type: str,
            dataset_type: str,
            features: List[np.ndarray]) -> List[List[float]]:
    if dataset_type == 'regression':
        preds = model.predict(features)

        if len(preds.shape) == 1:
            preds = [[pred] for pred in preds]
    elif dataset_type == 'classification':
        if model_type == 'random_forest':
            preds = model.predict_proba(features)

            if type(preds) == list:
                # Multiple tasks
                num_tasks, num_preds = len(preds), len(preds[0])
                preds = [[preds[i][j, 1] for i in range(num_tasks)] for j in range(num_preds)]
            else:
                # One task
                preds = [[preds[i, 1]] for i in range(len(preds))]
        elif model_type == 'svm':
            preds = model.decision_function(features)
            preds = [[pred] for pred in preds]
        else:
            raise ValueError(f'Model type "{model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{dataset_type}" not supported')

    return preds


def single_task_sklearn(model,
                        train_data: MoleculeDataset,
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

        model.fit(train_features, train_targets)

        test_preds = predict(
            model=model,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            features=test_features
        )
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


def multi_task_sklearn(model,
                       train_data: MoleculeDataset,
                       test_data: MoleculeDataset,
                       metric_func: Callable,
                       args: Namespace,
                       logger: Logger = None) -> List[float]:
    num_tasks = train_data.num_tasks()

    train_targets = train_data.targets()
    if train_data.num_tasks() == 1:
        train_targets = [targets[0] for targets in train_targets]

    # Train
    model.fit(train_data.features(), train_targets)

    # Save model
    with open(os.path.join(args.save_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    test_preds = predict(
        model=model,
        model_type=args.model_type,
        dataset_type=args.dataset_type,
        features=test_data.features()
    )

    scores = evaluate_predictions(
        preds=test_preds,
        targets=test_data.targets(),
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=args.dataset_type,
        logger=logger
    )

    return scores


def run_sklearn(args: Namespace, logger: Logger = None) -> List[float]:
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    metric_func = get_metric_func(args.metric)

    debug('Loading data')
    data = get_data(path=args.data_path)

    if args.model_type == 'svm' and data.num_tasks() != 1:
        raise ValueError(f'SVM can only handle single-task data but found {data.num_tasks()} tasks')

    debug(f'Splitting data with seed {args.seed}')
    # Need to have val set so that train and test sets are the same as when doing MPN
    train_data, _, test_data = split_data(
        data=data,
        split_type=args.split_type,
        seed=args.seed,
        sizes=args.split_sizes,
        args=args
    )

    debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | test size = {len(test_data):,}')

    debug('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for dataset in [train_data, test_data]:
        for datapoint in tqdm(dataset, total=len(dataset)):
            datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    debug('Building model')
    if args.dataset_type == 'regression':
        if args.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1)
        elif args.model_type == 'svm':
            model = SVR()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    elif args.dataset_type == 'classification':
        if args.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=args.num_trees, n_jobs=-1, class_weight=args.class_weight)
        elif args.model_type == 'svm':
            model = SVC()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported')

    debug(model)

    debug('Training')
    if args.single_task:
        scores = single_task_sklearn(
            model=model,
            train_data=train_data,
            test_data=test_data,
            metric_func=metric_func,
            args=args,
            logger=logger
        )
    else:
        scores = multi_task_sklearn(
            model=model,
            train_data=train_data,
            test_data=test_data,
            metric_func=metric_func,
            args=args,
            logger=logger
        )

    info(f'Test {args.metric} = {np.nanmean(scores)}')

    return scores


def cross_validate_sklearn(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    info = logger.info if logger is not None else print
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        model_scores = run_sklearn(args, logger)
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
