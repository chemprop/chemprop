from logging import Logger
import os
import pickle
from typing import Dict, List, Union
from copy import deepcopy

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, SGDRegressor
from tqdm import trange, tqdm

from chemprop.args import SklearnTrainArgs
from chemprop.data import MoleculeDataset, split_data, get_task_names, get_data
from chemprop.features import get_features_generator
from chemprop.train import cross_validate, evaluate_predictions
from chemprop.utils import save_smiles_splits


def predict(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
            model_type: str,
            dataset_type: str,
            features: List[np.ndarray]) -> List[List[float]]:
    """
    Predicts using a scikit-learn model.

    :param model: The trained scikit-learn model to make predictions with.
    :param model_type: The type of model.
    :param dataset_type: The type of dataset.
    :param features: The data features used as input for the model.
    :return: A list of lists of floats containing the predicted values.
    """
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

def impute_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                   train_data: MoleculeDataset,
                   args: SklearnTrainArgs,
                   logger: Logger = None,
                   threshold: float = 0.5) -> List[float]:
    """
    Trains a single-task scikit-learn model, meaning a separate model is trained for each task.

    This is necessary if some tasks have None (unknown) values.

    :param model: The scikit-learn model to train.
    :param train_data: The training data.
    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 training the scikit-learn model.
    :param logger: A logger to record output.
    :param theshold: Threshold for classification tasks.
    :return: A list of list of target values.
    """
    num_tasks = train_data.num_tasks()
    new_targets=deepcopy(train_data.targets())
    
    if logger is not None:
        debug = logger.debug
    else:
        debug = print
        
    debug('Imputation')
    
    for task_num in trange(num_tasks):
        impute_train_features = [features for features, targets in zip(train_data.features(), train_data.targets()) if targets[task_num] is None]
        if len(impute_train_features) > 0:
            train_features, train_targets = zip(*[(features, targets[task_num])
                                              for features, targets in zip(train_data.features(), train_data.targets())
                                              if targets[task_num] is not None])
            if args.impute_mode == 'single_task':
                model.fit(train_features, train_targets)
                impute_train_preds = predict(
                    model=model,
                    model_type=args.model_type,
                    dataset_type=args.dataset_type,
                    features=impute_train_features
                )
                impute_train_preds = [pred[0] for pred in impute_train_preds]
            elif args.impute_mode == 'median' and args.dataset_type == 'regression':
                impute_train_preds = [np.median(train_targets)] * len(new_targets)
            elif args.impute_mode == 'mean' and args.dataset_type == 'regression':
                impute_train_preds = [np.mean(train_targets)] * len(new_targets)
            elif args.impute_mode == 'frequent' and args.dataset_type == 'classification':
                impute_train_preds = [np.argmax(np.bincount(train_targets))] * len(new_targets)
            elif args.impute_mode == 'linear' and args.dataset_type == 'regression':
                reg = SGDRegressor(alpha=0.01).fit(train_features, train_targets)
                impute_train_preds = reg.predict(impute_train_features)
            elif args.impute_mode == 'linear' and args.dataset_type == 'classification':
                cls = SGDClassifier().fit(train_features, train_targets)
                impute_train_preds = cls.predict(impute_train_features)
            else:
                raise ValueError("Invalid combination of imputation mode and dataset type.")   

            #Replace targets
            ctr = 0
            for i in range(len(new_targets)):
                if new_targets[i][task_num] is None:
                    value = impute_train_preds[ctr]
                    if args.dataset_type == 'classification':
                        value = int(value > threshold)
                    new_targets[i][task_num] = value
                    ctr += 1
                    
    return new_targets


def single_task_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                        train_data: MoleculeDataset,
                        test_data: MoleculeDataset,
                        metrics: List[str],
                        args: SklearnTrainArgs,
                        logger: Logger = None) -> List[float]:
    """
    Trains a single-task scikit-learn model, meaning a separate model is trained for each task.

    This is necessary if some tasks have None (unknown) values.

    :param model: The scikit-learn model to train.
    :param train_data: The training data.
    :param test_data: The test data.
    :param metrics: A list of names of metric functions.
    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 training the scikit-learn model.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    scores = {}
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
            metrics=metrics,
            dataset_type=args.dataset_type,
            logger=logger
        )
        for metric in metrics:
            if metric not in scores:
                scores[metric] = []
            scores[metric].append(score[metric][0])

    return scores


def multi_task_sklearn(model: Union[RandomForestRegressor, RandomForestClassifier, SVR, SVC],
                       train_data: MoleculeDataset,
                       test_data: MoleculeDataset,
                       metrics: List[str],
                       args: SklearnTrainArgs,
                       logger: Logger = None) -> Dict[str, List[float]]:
    """
    Trains a multi-task scikit-learn model, meaning one model is trained simultaneously on all tasks.

    This is only possible if none of the tasks have None (unknown) values.

    :param model: The scikit-learn model to train.
    :param train_data: The training data.
    :param test_data: The test data.
    :param metrics: A list of names of metric functions.
    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 training the scikit-learn model.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    num_tasks = train_data.num_tasks()

    train_targets = train_data.targets()

    if args.impute_mode:
        train_targets = impute_sklearn(model=model,
                                       train_data=train_data,
                                       args=args,
                                       logger=logger)
    elif any(None in sublist for sublist in train_targets):
        raise ValueError("Missing target values not tolerated for multi-task sklearn models." 
                         "Use either --single_task to train multiple single-task models or impute"
                         " targets via --impute_mode  <model/linear/median/mean/frequent>.")
        
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
        metrics=metrics,
        dataset_type=args.dataset_type,
        logger=logger
    )

    return scores


def run_sklearn(args: SklearnTrainArgs,
                data: MoleculeDataset,
                logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a scikit-learn model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.SklearnTrainArgs` object containing arguments for
                 loading data and training the scikit-learn model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`metrics` to a list of values for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug('Loading data')
    data = get_data(path=args.data_path,
                    features_path=args.features_path,
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    logger=logger)
    args.task_names = get_task_names(path=args.data_path,
                                     smiles_columns=args.smiles_columns,
                                     target_columns=args.target_columns,
                                     ignore_columns=args.ignore_columns)

    if args.model_type == 'svm' and data.num_tasks() != 1:
        raise ValueError(f'SVM can only handle single-task data but found {data.num_tasks()} tasks')

    debug(f'Splitting data with seed {args.seed}')
    # Need to have val set so that train and test sets are the same as when doing MPN
    train_data, _, test_data = split_data(
        data=data,
        split_type=args.split_type,
        seed=args.seed,
        sizes=args.split_sizes,
        num_folds=args.num_folds,
        args=args,
        logger=logger
    )

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger
        )

    debug(f'Total size = {len(data):,} | train size = {len(train_data):,} | test size = {len(test_data):,}')

    debug('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for dataset in [train_data, test_data]:
        for datapoint in tqdm(dataset, total=len(dataset)):
            for s in datapoint.smiles:
                datapoint.extend_features(morgan_fingerprint(mol=s, radius=args.radius, num_bits=args.num_bits))

    debug('Building model')
    if args.dataset_type == 'regression':
        if args.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=args.num_trees, n_jobs=-1, random_state=args.seed)
        elif args.model_type == 'svm':
            model = SVR()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    elif args.dataset_type == 'classification':
        if args.model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=args.num_trees, n_jobs=-1, class_weight=args.class_weight, random_state=args.seed)
        elif args.model_type == 'svm':
            model = SVC()
        else:
            raise ValueError(f'Model type "{args.model_type}" not supported')
    else:
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported')

    debug(model)

    model.train_args = args.as_dict()

    debug('Training')
    if args.single_task:
        scores = single_task_sklearn(
            model=model,
            train_data=train_data,
            test_data=test_data,
            metrics=args.metrics,
            args=args,
            logger=logger
        )
    else:
        scores = multi_task_sklearn(
            model=model,
            train_data=train_data,
            test_data=test_data,
            metrics=args.metrics,
            args=args,
            logger=logger
        )

    for metric in args.metrics:
        info(f'Test {metric} = {np.nanmean(scores[metric])}')

    return scores


def sklearn_train() -> None:
    """Parses scikit-learn training arguments and trains a scikit-learn model.

    This is the entry point for the command line command :code:`sklearn_train`.
    """
    cross_validate(args=SklearnTrainArgs().parse_args(), train_func=run_sklearn)
