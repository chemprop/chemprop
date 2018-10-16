import argparse
import os

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm


def morgan_fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_vect, fp)

    return fp


def save_fingerprints(data_path: str, save_path: str):
    data = []
    with open(data_path) as f:
        header = f.readline().strip().split(',')
        for line in f:
            vals = line.strip("\r\n ").split(',')
            smiles = vals[0]
            vals = vals[1:]
            data.append((smiles, vals))

    save_dir = os.path.dirname(save_path)
    if save_dir != '' and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_path, 'w') as f:
        f.write(header[0] + ',fingerprint,' + ','.join(header[1:]) + '\n')

        for smiles, vals in tqdm(data, total=len(data)):
            fp = morgan_fingerprint(smiles)
            f.write(smiles + ',' + fp + ',' + ','.join(vals) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path to data CSV file')
    parser.add_argument('--save_path', type=str, help='Path to CSV with morgan fingerprints')
    args = parser.parse_args()

    save_fingerprints(args.data_path, args.save_path)
