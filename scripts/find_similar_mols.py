"""Script to find the molecules in the training set which are most similar to each molecule in the test set."""

from collections import OrderedDict
import csv
import os
import sys
from typing import List
from typing_extensions import Literal

from rdkit import DataStructs
from rdkit import Chem

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_data_from_smiles, get_smiles, MoleculeDataLoader
from chemprop.features import morgan_binary_features_generator
from chemprop.models import MoleculeModel
from chemprop.utils import load_checkpoint, makedirs
from chemprop.train import model_fingerprint


class Args(Tap):
    test_path: str  # Path to CSV file with test set of molecules
    train_path: str  # Path to CSV file with train set of molecules
    save_path: str  # Path to CSV file where similar molecules will be saved
    distance_measure: Literal['embedding', 'morgan', 'tanimoto'] = 'embedding'  # Distance measure to use to find nearest neighbors in train set
    checkpoint_path: str = None  # Path to .pt file containing a model checkpoint (only needed for distance_measure == "embedding")
    num_neighbors: int = 5  # Number of neighbors to search for each molecule
    batch_size: int = 50  # Batch size when making predictions
    smiles_column: str = None # Columns in dataset CSV file containing SMILES
    num_workers: int = 8 # Number of workers used to build batches.


def find_similar_mols(test_smiles: List[str],
                      train_smiles: List[str],
                      distance_measure: str,
                      model: MoleculeModel = None,
                      num_neighbors: int = None,
                      batch_size: int = 50) -> List[OrderedDict]:
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.

    :param test_smiles: A list of test SMILES strings.
    :param train_smiles: A list of train SMILES strings.
    :param model: A trained MoleculeModel (only needed for distance_measure == 'embedding').
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """
    test_data = get_data_from_smiles(smiles=[[smiles] for smiles in test_smiles])
    train_data = get_data_from_smiles(smiles=[[smiles] for smiles in train_smiles])
    train_smiles_set = set(train_smiles)

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=args.num_workers
    )
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=args.num_workers
    )

    print(f'Computing {distance_measure} vectors')
    if distance_measure == 'embedding':
        assert model is not None
        test_vecs = np.array(model_fingerprint(model=model, data_loader=test_data_loader, fingerprint_type='last_FFN'))
        train_vecs = np.array(model_fingerprint(model=model, data_loader=train_data_loader, fingerprint_type='last_FFN'))
        metric = 'cosine'
    elif distance_measure == 'morgan':
        test_vecs = np.array([morgan_binary_features_generator(smiles) for smiles in tqdm(test_smiles, total=len(test_smiles))])
        train_vecs = np.array([morgan_binary_features_generator(smiles) for smiles in tqdm(train_smiles, total=len(train_smiles))])
        metric = 'jaccard'
    elif distance_measure == 'tanimoto':
        # Generate RDKit topological fingerprints
        test_fps = [Chem.RDKFingerprint(m.mol[0]) for m in tqdm(test_data)]
        train_fps = [Chem.RDKFingerprint(m.mol[0]) for m in tqdm(train_data)]

        # Compute pairwise similarity
        print('Computing distances')
        similarity = np.zeros([len(test_fps), len(train_fps)])
        for (x, y), _ in np.ndenumerate(similarity):
            similarity[x, y] = DataStructs.FingerprintSimilarity(test_fps[x], train_fps[y])

        # Convert the tanimoto similarity to a distance
        distances = 1 - similarity
        metric = 'tanimoto'
    else:
        raise ValueError(f'Distance measure "{distance_measure}" not supported.')

    if distance_measure in ('embedding', 'morgan'):
        print('Computing distances')
        distances = cdist(test_vecs, train_vecs, metric=metric)

    print('Finding neighbors')
    neighbors = []
    for test_index, test_smile in enumerate(test_smiles):
        # Find the num_neighbors molecules in the training set which are most similar to the test molecule
        nearest_train_indices = np.argsort(distances[test_index])[:num_neighbors]

        # Build dictionary with distance info
        neighbor = OrderedDict()
        neighbor['test_smiles'] = test_smile
        neighbor['test_in_train'] = test_smile in train_smiles_set

        for i, train_index in enumerate(nearest_train_indices):
            neighbor[f'train_{i + 1}_smiles'] = train_smiles[train_index]
            neighbor[f'train_{i + 1}_{distance_measure}_{metric}_distance'] = distances[test_index][train_index]

        neighbors.append(neighbor)

    return neighbors


def find_similar_mols_from_file(test_path: str,
                                train_path: str,
                                distance_measure: str,
                                checkpoint_path: str = None,
                                num_neighbors: int = -1,
                                batch_size: int = 50,
                                smiles_column: str = None) -> List[OrderedDict]:
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.
    Loads molecules and model from file.

    :param test_path: Path to a CSV file containing test SMILES.
    :param train_path: Path to a CSV file containing train SMILES.
    :param checkpoint_path: Path to a .pt model checkpoint file (only needed for distance_measure == 'embedding').
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """
    print('Loading data')
    test_smiles, train_smiles = get_smiles(test_path, flatten=True, smiles_columns=smiles_column), get_smiles(train_path, flatten=True, smiles_columns=smiles_column)

    if checkpoint_path is not None:
        print('Loading model')
        model = load_checkpoint(checkpoint_path)
    else:
        model = None

    return find_similar_mols(
        test_smiles=test_smiles,
        train_smiles=train_smiles,
        distance_measure=distance_measure,
        model=model,
        num_neighbors=num_neighbors,
        batch_size=batch_size
    )


def save_similar_mols(test_path: str,
                      train_path: str,
                      save_path: str,
                      distance_measure: str,
                      checkpoint_path: str = None,
                      num_neighbors: int = None,
                      batch_size: int = 50,
                      smiles_column: str = None):
    """
    For each test molecule, finds the N most similar training molecules according to some distance measure.
    Loads molecules and model from file and saves results to file.

    :param test_path: Path to a CSV file containing test SMILES.
    :param train_path: Path to a CSV file containing train SMILES.
    :param checkpoint_path: Path to a .pt model checkpoint file (only needed for distance_measure == 'embedding').
    :param save_path: Path to a CSV file where the results will be saved.
    :param distance_measure: The distance measure to use to determine nearest neighbors.
    :param num_neighbors: The number of nearest training molecules to find for each test molecule.
    :param batch_size: Batch size.
    :return: A list of OrderedDicts containing the test smiles, the num_neighbors nearest training smiles,
    and other relevant distance info.
    """

    # Find similar molecules
    similar_mols = find_similar_mols_from_file(
        test_path=test_path,
        train_path=train_path,
        checkpoint_path=checkpoint_path,
        distance_measure=distance_measure,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        smiles_column=smiles_column,
    )

    # Save results
    makedirs(save_path, isfile=True)

    with open(save_path, 'w', newline="") as f:
        writer = csv.DictWriter(f, fieldnames=similar_mols[0].keys())
        writer.writeheader()
        for row in similar_mols:
            writer.writerow(row)


if __name__ == '__main__':
    args = Args().parse_args()

    save_similar_mols(
        test_path=args.test_path,
        train_path=args.train_path,
        save_path=args.save_path,
        distance_measure=args.distance_measure,
        checkpoint_path=args.checkpoint_path,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        smiles_column=args.smiles_column,
    )
