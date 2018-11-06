from itertools import product
from typing import List

import math
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from .scaffold import scaffold_to_smiles


def scaffold_similarity(smiles_1: List[str], smiles_2: List[str]):
    """
    Determines the similarity between the scaffolds of two lists of smiles strings.

    :param smiles_1: A list of smiles strings.
    :param smiles_2: A list of smiles strings.
    """
    # Get scaffolds
    scaffold_to_smiles_1 = scaffold_to_smiles(smiles_1)
    scaffold_to_smiles_2 = scaffold_to_smiles(smiles_2)

    scaffolds_1, smiles_sets_1 = zip(*scaffold_to_smiles_1.items())
    scaffolds_2, smiles_sets_2 = zip(*scaffold_to_smiles_2.items())

    smiles_to_scaffold = {smiles: scaffold for scaffold, smiles_set in scaffold_to_smiles_1.items() for smiles in smiles_set}
    smiles_to_scaffold.update({smiles: scaffold for scaffold, smiles_set in scaffold_to_smiles_2.items() for smiles in smiles_set})


    # Determine similarity
    scaffolds_1, scaffolds_2 = set(scaffolds_1), set(scaffolds_2)
    smiles_1, smiles_2 = set(smiles_1), set(smiles_2)

    all_scaffolds = scaffolds_1 | scaffolds_2
    all_smiles = smiles_1 | smiles_2

    scaffolds_intersection = scaffolds_1 & scaffolds_2
    # smiles_intersection is smiles with a scaffold that appears in both datasets
    smiles_intersection = {smiles for smiles in all_smiles if smiles_to_scaffold[smiles] in scaffolds_intersection}

    smiles_in_1_with_scaffold_in_2 = {smiles for smiles in smiles_1 if smiles_to_scaffold[smiles] in scaffolds_2}
    smiles_in_2_with_scaffold_in_1 = {smiles for smiles in smiles_2 if smiles_to_scaffold[smiles] in scaffolds_1}

    sizes_1 = np.array([len(smiles_set) for smiles_set in smiles_sets_1])
    sizes_2 = np.array([len(smiles_set) for smiles_set in smiles_sets_2])

    # Print results
    print()
    print('Number of molecules = {:,}'.format(len(all_smiles)))
    print('Number of scaffolds = {:,}'.format(len(all_scaffolds)))
    print()
    print('Number of scaffolds in both datasets = {:,}'.format(len(scaffolds_intersection)))
    print('Scaffold intersection over union = {:.4f}'.format(len(scaffolds_intersection) / len(all_scaffolds)))
    print()
    print('Number of molecules with scaffold in both datasets = {:,}'.format(len(smiles_intersection)))
    print('Molecule intersection over union = {:.4f}'.format(len(smiles_intersection) / len(all_smiles)))
    print()
    print('Number of molecules in dataset 1 = {:,}'.format(np.sum(sizes_1)))
    print('Number of scaffolds in dataset 1 = {:,}'.format(len(scaffolds_1)))
    print()
    print('Number of molecules in dataset 2 = {:,}'.format(np.sum(sizes_2)))
    print('Number of scaffolds in dataset 2 = {:,}'.format(len(scaffolds_2)))
    print()
    print('Percent of scaffolds in dataset 1 which are also in dataset 2 = {:.2f}%'.format(100 * len(scaffolds_intersection) / len(scaffolds_1)))
    print('Percent of scaffolds in dataset 2 which are also in dataset 1 = {:.2f}%'.format(100 * len(scaffolds_intersection) / len(scaffolds_2)))
    print()
    print('Number of molecules in dataset 1 with scaffolds in dataset 2 = {:,}'.format(len(smiles_in_1_with_scaffold_in_2)))
    print('Percent of molecules in dataset 1 with scaffolds in dataset 2 = {:.2f}%'.format(100 * len(smiles_in_1_with_scaffold_in_2) / len(smiles_1)))
    print()
    print('Number of molecules in dataset 2 with scaffolds in dataset 1 = {:,}'.format(len(smiles_in_2_with_scaffold_in_1)))
    print('Percent of molecules in dataset 2 with scaffolds in dataset 1 = {:.2f}%'.format(100 * len(smiles_in_2_with_scaffold_in_1) / len(smiles_2)))
    print()
    print('Average number of molecules per scaffold in dataset 1 = {:.4f} +/- {:.4f}'.format(np.mean(sizes_1), np.std(sizes_1)))
    print('Percentiles for molecules per scaffold in dataset 1')
    print(' | '.join(['{}% = {:,}'.format(i, int(np.percentile(sizes_1, i))) for i in range(0, 101, 10)]))
    print()
    print('Average number of molecules per scaffold in dataset 2 = {:.4f} +/- {:.4f}'.format(np.mean(sizes_2), np.std(sizes_2)))
    print('Percentiles for molecules per scaffold in dataset 2')
    print(' | '.join(['{}% = {:,}'.format(i, int(np.percentile(sizes_2, i))) for i in range(0, 101, 10)]))


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
    num_pairs = len(smiles_1) * len(smiles_2)

    # Sample to improve speed
    if sample_rate < 1.0:
        sample_num_pairs = sample_rate * num_pairs
        sample_size = math.ceil(math.sqrt(sample_num_pairs))
        sample_smiles_1 = np.random.choice(smiles_1, size=sample_size, replace=True)
        sample_smiles_2 = np.random.choice(smiles_2, size=sample_size, replace=True)
    else:
        sample_smiles_1, sample_smiles_2 = smiles_1, smiles_2

    sample_num_pairs = len(sample_smiles_1) * len(sample_smiles_2)

    for smile_1, smile_2 in tqdm(product(sample_smiles_1, sample_smiles_2), total=sample_num_pairs):
        mol_1, mol_2 = Chem.MolFromSmiles(smile_1), Chem.MolFromSmiles(smile_2)
        fp_1, fp_2 = AllChem.GetMorganFingerprint(mol_1, radius), AllChem.GetMorganFingerprint(mol_2, radius)
        similarity = DataStructs.DiceSimilarity(fp_1, fp_2)
        similarities.append(similarity)
    similarities = np.array(similarities)

    # Print results
    print()
    print('Average dice similarity = {:.4f} +/- {:.4f}'.format(np.mean(similarities), np.std(similarities)))
    print('Minimum dice similarity = {:.4f}'.format(np.min(similarities)))
    print('Maximum dice similarity = {:.4f}'.format(np.max(similarities)))
    print()
    print('Percentiles for dice similarity')
    print(' | '.join(['{}% = {:.4f}'.format(i, np.percentile(similarities, i)) for i in range(0, 101, 10)]))
