:py:mod:`featurizers.atom`
==========================

.. py:module:: featurizers.atom


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.atom.AtomFeaturizer




.. py:class:: AtomFeaturizer


   Bases: :py:obj:`chemprop.v2.featurizers.protos.AtomFeaturizerProto`

   An :class:`AtomFeaturizer` featurizes atoms based on the following attributes:

   * atomic number
   * degree
   * formal charge
   * chiral tag
   * number of hydrogens
   * hybridization
   * aromaticity
   * mass

   The feature vectors produced by this featurizer have the following (general) signature:

   +---------------------+-----------------+--------------+
   | slice [start, stop) | subfeature      | unknown pad? |
   +=====================+=================+==============+
   | 0-101               | atomic number   | Y            |
   +---------------------+-----------------+--------------+
   | 101-108             | degree          | Y            |
   +---------------------+-----------------+--------------+
   | 108-114             | formal charge   | Y            |
   +---------------------+-----------------+--------------+
   | 114-119             | chiral tag      | Y            |
   +---------------------+-----------------+--------------+
   | 119-125             | # Hs            | Y            |
   +---------------------+-----------------+--------------+
   | 125-131             | hybridization   | Y            |
   +---------------------+-----------------+--------------+
   | 131-132             | aromatic?       | N            |
   +---------------------+-----------------+--------------+
   | 132-133             | mass            | N            |
   +---------------------+-----------------+--------------+

   NOTE: the above signature only applies for the default arguments, as the each slice (save for
   the final two) can increase in size depending on the input arguments.

   .. py:attribute:: max_atomic_num
      :type: dataclasses.InitVar[int]
      :value: 100

      

   .. py:attribute:: degrees
      :type: Sequence[int]

      

   .. py:attribute:: formal_charges
      :type: Sequence[int]

      

   .. py:attribute:: chiral_tags
      :type: Sequence[int]

      

   .. py:attribute:: num_Hs
      :type: Sequence[int]

      

   .. py:attribute:: hybridizations
      :type: Sequence[rdkit.Chem.rdchem.HybridizationType]

      

   .. py:method:: __post_init__(max_atomic_num = 100)


   .. py:method:: __len__()

      the length of an atomic feature vector


   .. py:method:: __call__(a)

      featurize the atom ``a``


   .. py:method:: num_only(a)

      featurize the atom by setting only the atomic number bit



