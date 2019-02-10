from typing import Callable, Union

import numpy as np
from rdkit import Chem

try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    from descriptastorus.descriptors import DescriptorGenerator

    class RDKit2D_variant(DescriptorGenerator):
        """Computes all RDKit Descriptors"""
        NAME = "RDKit2D"
        def __init__(self, properties=rdDescriptors.RDKIT_PROPS[rdDescriptors.CURRENT_VERSION],short_name=None):
            if properties and short_name is not None:
                  self.NAME=self.NAME+'-'+short_name.lower()
            else:
                raise ValueError("If a properties list is provided a short_name has to be provided, too.")
            DescriptorGenerator.__init__(self)
            # specify names and numpy types for all columns
            if not properties:
                self.columns = [ (name, np.float64) for name,func in sorted(Descriptors.descList) ]
            else:
                columns = self.columns
                failed = []
                for name in properties:
                    if name in sorted(rdDescriptors.FUNCS):
                        columns.append((name, np.float64))
                    else:
                        logging.error("Unable to find specified property %s"%name)
                        failed.append(name)
                if failed:
                    raise ValueError("%s: Failed to initialize: unable to find specified properties:\n\t%s"%(
                        self.__class__.__name__,
                        "\n\t".join(failed)))
        def calculateMol(self, m, smiles, internalParsing=False):
            res = [ applyFunc(name, m) for name, _ in self.columns ]
            return res
        
    class MorganCounts_variant(DescriptorGenerator):
        """Computes Morgan3 bitvector counts"""
        NAME = "Morgan%s"
        def __init__(self, radius=3, nbits=2048, count=True):
            self.count = count
            if count:
                ctype="Counts"
            else:
                ctype="Bits"
            if radius == 3 and nbits == 2048:
                self.NAME = (self.NAME % "3")+ctype
            else:
                self.NAME = (self.NAME%radius)+ctype+"-%s"%nbits
                
            DescriptorGenerator.__init__(self)
            # specify names and numpy types for all columns+ctype
            self.radius = radius
            self.nbits = nbits
            morgan = [("m3-%d"%d, np.uint8) for d in range(nbits)]
            self.columns += morgan

        def calculateMol(self, m, smiles, internalParsing=False):
            if self.count:
                counts = list(rd.GetHashedMorganFingerprint(m,radius=self.radius, nBits=self.nbits))
            else:
                counts = list(rd.GetMorganFingerprintAsBitVect(m,radius=self.radius, nBits=self.nbits))
            counts = [ clip(x,smiles) for x in counts ]
            return counts        

except ImportError:
    raise ImportError('Descriptastorus not available. Please install it for rdkit descriptors.')
