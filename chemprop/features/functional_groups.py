from argparse import Namespace
import numpy as np
from typing import List, Union

from rdkit import Chem


def get_num_functional_groups(args: Namespace):
    with open(args.functional_group_smarts, 'r') as f:
        count = len(f.readlines())
    return count


class FunctionalGroupFeaturizer:
    """
    Class for extracting feature vector of indicators for atoms being parts of functional groups. 
    """
    def __init__(self, args: Namespace):
        self.smarts = []
        with open(args.functional_group_smarts, 'r') as f:
            for line in f:
                self.smarts.append(Chem.MolFromSmarts(line.strip()))
    
    def featurize(self, smiles: Union[Chem.Mol, str]) -> List[List[int]]:
        """
        Given a molecule in SMILES form, return a feature vector of indicators for each atom,
        indicating whether the atom is part of each functional group. 
        Can also directly accept a Chem molecule.
        Searches through the functional groups given in smarts.txt. 

        :param smiles: A smiles string representing a molecule.
        :return: Numpy array of shape num_atoms x num_features (num functional groups)
        """
        if type(smiles) == str:
            mol = Chem.MolFromSmiles(smiles)  # turns out rdkit knows to match even without adding Hs
        else:
            mol = smiles
        features = np.zeros((mol.GetNumAtoms(), len(self.smarts)))  # num atoms (without Hs) x num features
        for i, smarts in enumerate(self.smarts):
            for group in mol.GetSubstructMatches(smarts):
                for idx in group:
                    features[idx][i] = 1

        return features.tolist()


if __name__ == '__main__':
    featurizer = FunctionalGroupFeaturizer()
    features = featurizer.featurize('C(#N)C(=O)C#N')
