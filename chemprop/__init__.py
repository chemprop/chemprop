import re

import chemprop.data
import chemprop.features
import chemprop.models
import chemprop.train

import chemprop.nn_utils
import chemprop.parsing
import chemprop.utils

try:
    from descriptastorus.descriptors import DescriptorGenerator
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
    from chemprop.features.rdkit_features import MorganCounts_variant, RDKit2D_variant

    fpSize=2048
    rdDescriptors.AtomPairCounts(nbits=fpSize)
    rdDescriptors.MorganCounts(nbits=fpSize)
    MorganCounts_variant(radius=2,nbits=fpSize,count=True)
    MorganCounts_variant(radius=2,nbits=fpSize,count=False)
    rdDescriptors.ChiralMorganCounts(nbits=fpSize)
    rdDescriptors.FeatureMorganCounts(nbits=fpSize)
    rdDescriptors.AtomPairCounts(nbits=fpSize)
    rdDescriptors.RDKitFPBits(nbits=fpSize)

    regex = re.compile(r'^PEOE')
    filtered = [i for i in rdDescriptors.RDKIT_PROPS[rdDescriptors.CURRENT_VERSION] if regex.match(i)]
    RDKit2D_variant(properties=filtered,short_name="peoe")

except ImportError:
    raise ImportError('Descriptastorus not available. Please install it for rdkit descriptors.')
