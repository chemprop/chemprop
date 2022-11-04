"""Computes and saves molecular features for a dataset."""

from multiprocessing import Pool
import os
import shutil
import sys
from typing import List, Tuple

from tqdm import tqdm
from tap import Tap  # pip install typed-argument-parser (https://github.com/swansonk14/typed-argument-parser)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from chemprop.data import get_smiles
from chemprop.features import get_available_features_generators, get_features_generator, load_features, save_features
from chemprop.utils import makedirs


class Args(Tap):
    data_path: str  # Path to data CSV
    smiles_column: str = None  # Name of the column containing SMILES strings. By default, uses the first column.
    features_generator: str = 'rdkit_2d_normalized'  # Type of features to generate
    save_path: str  # Path to .npz file where features will be saved as a compressed numpy archive
    save_frequency: int = 10000  # Frequency with which to save the features
    restart: bool = False  # Whether to not load partially complete featurization and instead start from scratch
    sequential: bool = False  # Whether to run sequentially rather than in parallel

    def configure(self) -> None:
        self.add_argument('--features_generator', choices=get_available_features_generators())


def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.

    Assumes temporary files are named in order 0.npz, 1.npz, ...

    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def generate_and_save_features(args: Args):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

    :param args: Arguments.
    """
    # Create directory for save_path
    makedirs(args.save_path, isfile=True)

    # Get data and features function
    smiles = get_smiles(path=args.data_path, smiles_columns=args.smiles_column, flatten=True)
    features_generator = get_features_generator(args.features_generator)
    temp_save_dir = args.save_path + '_temp'

    # Load partially complete data
    if args.restart:
        if os.path.exists(args.save_path):
            os.remove(args.save_path)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(args.save_path):
            raise ValueError(f'"{args.save_path}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
        features, temp_num = [], 0

    # Build features map function
    smiles = smiles[len(features):]  # restrict to data for which features have not been computed yet

    if args.sequential:
        features_map = map(features_generator, smiles)
    else:
        features_map = Pool().imap(features_generator, smiles)

    # Get features
    temp_features = []
    for i, feats in tqdm(enumerate(features_map), total=len(smiles)):
        temp_features.append(feats)

        # Save temporary features every save_frequency
        if (i > 0 and (i + 1) % args.save_frequency == 0) or i == len(smiles) - 1:
            save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        # Save all features
        save_features(args.save_path, features)

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')


if __name__ == '__main__':
    generate_and_save_features(Args().parse_args())
