import logging
from rdkit import Chem

try:
    from descriptastorus.descriptors import rdNormalizedDescriptors
    generator = rdNormalizedDescriptors.RDKit2DNormalized()

    def rdkit_2d_normalized_features(smiles: str):
        # the first element is true/false if the mol was properly computed
        if type(smiles) == str:
            return generator.process(smiles)[1:]
        
        else:
            # this is a bit of a waste, but the desciptastorus API is smiles
            #  based for normalization purposes
            return generator.process(Chem.MolToSmiles(smiles, isomericSmiles=True))[1:]


except ImportError:
    logging.getLogger(__name__).warning("descriptastorus is not available, normalized descriptors are not available")
    rdkit_2d_normalized_features = None





