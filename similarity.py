from argparse import ArgumentParser
from itertools import  product
from typing import List

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from scaffold import scaffold_to_smiles
from utils import get_data


def scaffold_similarity(smiles_1: List[str], smiles_2: List[str]):
    """
    Determines the similarity between the scaffolds of two lists of smiles strings.

    :param smiles_1: A list of smiles strings.
    :param smiles_2: A list of smiles strings.
    """
    # Get scaffolds
    scaffolds_1 = set(scaffold_to_smiles(smiles_1).keys())
    scaffolds_2 = set(scaffold_to_smiles(smiles_2).keys())

    # Determine similarity
    intersection = scaffolds_1 & scaffolds_2
    union = scaffolds_1 | scaffolds_2
    unique_1 = scaffolds_1 - scaffolds_2
    unique_2 = scaffolds_2 - scaffolds_1

    # Print results
    print('Number of scaffolds in dataset 1 = {:,}'.format(len(scaffolds_1)))
    print('Number of scaffolds in dataset 2 = {:,}'.format(len(scaffolds_2)))
    print('Number of scaffolds in intersection = {:,}'.format(len(intersection)))
    print('Number of scaffolds in union = {:,}'.format(len(union)))
    print('Intersection over union = {:.4f}'.format(len(intersection) / len(union)))
    print('Number of unique scaffolds in dataset 1 = {:,}'.format(len(unique_1)))
    print('Number of unique scaffolds in dataset 2 = {:,}'.format(len(unique_2)))


def morgan_similarity(smiles_1: List[str], smiles_2: List[str], radius: int, sample_rate: float):
    """
    Determines the similarity between the morgan fingerprints of two lists of smiles strings.

    :param smiles_1: A list of smiles strings.
    :param smiles_2: A list of smiles strings.
    :param radius: The radius of the morgan fingerprints.
    :param sample_rate: Rate at which to sample pairs of molecules for Morgan similarity (to reduce time).
    """
    # Compute similarities
    similarities = []
    for smile_1, smile_2 in tqdm(product(smiles_1, smiles_2), total=len(smiles_1) * len(smiles_2)):
        if np.random.rand() > sample_rate:
            continue
        
        mol_1, mol_2 = Chem.MolFromSmiles(smile_1), Chem.MolFromSmiles(smile_2)
        fp_1, fp_2 = AllChem.GetMorganFingerprint(mol_1, radius), AllChem.GetMorganFingerprint(mol_2, radius)
        similarity = DataStructs.DiceSimilarity(fp_1, fp_2)
        similarities.append(similarity)
    similarities = np.array(similarities)

    # Print results
    print('Average dice similarity = {:.4f} +/- {:.4f}'.format(np.mean(similarities), np.std(similarities)))
    print('Minimum dice similarity = {:.4f}'.format(np.min(similarities)))
    print('Maximum dice similarity = {:.4f}'.format(np.max(similarities)))
    print('Percentiles')
    print(' | '.join(['{}% = {:.4f}'.format(i, np.percentile(similarities, i)) for i in range(0, 101, 10)]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path_1', type=str, required=True,
                        help='Path to first data CSV file')
    parser.add_argument('--data_path_2', type=str, required=True,
                        help='Path to second data CSV file')
    parser.add_argument('--similarity_measure', type=str, required=True, choices=['scaffold', 'morgan'],
                        help='Similarity measure to use to compare the two datasets')
    parser.add_argument('--radius', type=int, default=6,
                        help='Radius of Morgan fingerprint')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Rate at which to sample pairs of molecules for Morgan similarity (to reduce time)')
    args = parser.parse_args()

    smiles_1 = get_data(args.data_path_1, smiles_only=True)
    smiles_2 = get_data(args.data_path_2, smiles_only=True)

    if args.similarity_measure == 'scaffold':
        scaffold_similarity(smiles_1, smiles_2)
    elif args.similarity_measure == 'morgan':
        morgan_similarity(smiles_1, smiles_2, args.radius, args.sample_rate)
    else:
        raise ValueError('Similarity measure "{}" not supported.'.format(args.similarity_measure))
