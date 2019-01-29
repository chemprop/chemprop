from typing import Callable, Union

import numpy as np
from rdkit import Chem

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

    def features_factory(generator) -> Callable[[Union[str, Chem.Mol]], np.ndarray]:
        """
        Uses a descriptastorus generator to construction a features generator function.

        :param generator: A descriptastorus generator.
        :return: A function mapping a molecule to a numpy array of features.
        """
        def features(mol: Union[str, Chem.Mol]) -> np.ndarray:
            """
            A function which generates features from a molecule.

            :param mol: A smiles string or an RDKit molecule.
            :return: A 1D numpy array of features.
            """
            if type(mol) == str:
                smiles = mol
            else:
                # This is a bit of a waste, but the desciptastorus API is smiles based for normalization purposes
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)

            # The first element is true/false if the mol was properly computed
            return generator.process(smiles)[1:]

        return features

    rdkit_2d_features = features_factory(rdDescriptors.RDKit2D())
    rdkit_2d_normalized_features = features_factory(rdNormalizedDescriptors.RDKit2DNormalized())
except ImportError:
    raise ImportError('Descriptastorus not available. Please install it for rdkit descriptors.')
