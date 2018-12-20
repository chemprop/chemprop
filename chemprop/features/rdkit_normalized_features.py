import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from rdkit import Chem

generator = rdNormalizedDescriptors.RDKit2DNormalized()

def rdkit_2d_normalized_features(smiles: str):
    # the first element is true/false if the mol was properly computed
    return generator.process(smiles)[1:]

