from abc import ABC, abstractmethod

from rdkit import Chem

class Featurizer(ABC):
    def __call__(self, mol):
        return self.featurize(mol)
        
    @abstractmethod
    def featurize(self, mol: Chem.Mol):
        pass