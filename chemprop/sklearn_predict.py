from argparse import Namespace
import csv
import pickle

import numpy as np
from tqdm import tqdm

from chemprop.data.utils import get_data, get_task_names
from chemprop.features import get_features_generator
from chemprop.sklearn_train import predict
from chemprop.utils import makedirs


def predict_sklearn(args: Namespace):
    print('Loading data')
    data = get_data(path=args.test_path)

    print('Computing morgan fingerprints')
    morgan_fingerprint = get_features_generator('morgan')
    for datapoint in tqdm(data, total=len(data)):
        datapoint.set_features(morgan_fingerprint(mol=datapoint.smiles, radius=args.radius, num_bits=args.num_bits))

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    sum_preds = np.zeros((len(data), args.num_tasks))

    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        with open(checkpoint_path, 'rb') as f:
            model = pickle.load(f)

        model_preds = predict(
            model=model,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            features=data.features()
        )
        sum_preds += np.array(model_preds)

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    print('Saving predictions')
    assert len(data) == len(avg_preds)
    makedirs(args.preds_path, isfile=True)

    with open(args.preds_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['smiles'] + get_task_names(args.test_path))

        for smiles, pred in zip(data.smiles(), avg_preds):
            writer.writerow([smiles] + pred)
