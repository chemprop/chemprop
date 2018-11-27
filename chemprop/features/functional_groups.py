from rdkit import Chem
import numpy as np


class FunctionalGroupFeaturizer:
    """
    Class for extracting feature vector of indicators for atoms being parts of functional groups. 
    """
    def __init__(self):
        self.smarts = []
        with open('smarts.txt', 'r') as f:
            for line in f:
                self.smarts.append(Chem.MolFromSmarts(line.strip()))
    
    def featurize(self, smiles):
        """
        Given a molecule in SMILES form, return a feature vector of indicators for each atom,
        indicating whether the atom is part of each functional group. 
        Searches through the functional groups given in smarts.txt. 

        :param smiles: A smiles string representing a molecule.
        :return: Numpy array of shape num_atoms x num_features (num functional groups)
        """
        mol = Chem.MolFromSmiles(smiles)  # turns out rdkit knows to match even without adding Hs
        features = np.zeros((mol.GetNumAtoms(), len(self.smarts)))  # num atoms (without Hs) x num features
        for i, smarts in enumerate(self.smarts):
            for group in mol.GetSubstructMatches(smarts):
                for idx in group:
                    features[idx][i] = 1
        return features


if __name__ == '__main__':
    featurizer = FunctionalGroupFeaturizer()
    features = featurizer.featurize('C(#N)C(=O)C#N')
