from typing import Callable, Union

import numpy as np
from rdkit import Chem

try:
    from mordred import Calculator, descriptors
    from descriptastorus.descriptors.rdNormalizedDescriptors import applyNormalizedFunc
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    from descriptastorus.descriptors import DescriptorGenerator

    class Mordred(DescriptorGenerator):
        """Computes all Mordred Descriptors"""
        NAME = "Mordred"
        
        def __init__(self):
            DescriptorGenerator.__init__(self)
            # specify names and numpy types for all columns
            calculator = Calculator(descriptors, ignore_3D=True)  # can't do 3D without sdf or mol file
            feature_names=[str(i) for i in calculator._descriptors ]
            self.columns = [ (name, np.float64) for name in feature_names ]
        
        def calculateMol(self, m, smiles, internalParsing=False):
            calculator = Calculator(descriptors, ignore_3D=True)
            return [float(f) for f in calculator(m)]

    class MordredNormalized(Mordred):
        NAME = "MordredNormalized"

        def calculateMol(self, m, smiles, internalParsing=False):
            res = [ applyNormalizedFunc(name, m) for name, _ in self.columns ]
            return res   

except ImportError:
    raise ImportError('Mordred not available. Please install it for mordred descriptors.')
