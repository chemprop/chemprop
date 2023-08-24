:py:mod:`featurizers.bond`
==========================

.. py:module:: featurizers.bond


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.bond.BondFeaturizerProto
   featurizers.bond.BondFeaturizer




.. py:class:: BondFeaturizerProto


   Bases: :py:obj:`Protocol`

   A `BondFeaturizerProto` calculates feature vectors of RDKit bonds

   .. py:method:: __len__() -> int

      the length of a bond feature vector


   .. py:method:: __call__(b: rdkit.Chem.rdchem.Bond) -> numpy.ndarray

      featurize the bond ``b``



.. py:class:: BondFeaturizer(bond_types: Sequence[rdkit.Chem.rdchem.BondType] | None = None, stereos: Sequence[int] | None = None)


   Bases: :py:obj:`BondFeaturizerProto`, :py:obj:`chemprop.v2.featurizers.utils.MultiHotFeaturizerMixin`

   A `BondFeaturizer` generates multihot featurizations of RDKit bonds

   The featurizations produced by this featurizer have the following (general) signature:

   +---------------------+-----------------+--------------+
   | slice [start, stop) | subfeature      | unknown pad? |
   +=====================+=================+==============+
   | 0-1                 | null?           | N            |
   +---------------------+-----------------+--------------+
   | 1-5                 | bond type       | N            |
   +---------------------+-----------------+--------------+
   | 5-6                 | conjugated?     | N            |
   +---------------------+-----------------+--------------+
   | 6-8                 | in ring?        | N            |
   +---------------------+-----------------+--------------+
   | 7-14                | stereochemistry | Y            |
   +---------------------+-----------------+--------------+

   **NOTE**: the above signature only applies for the default arguments, as the bond type and
   sterochemistry slices can increase in size depending on the input arguments.

   :param bond_types: the known bond types
   :type bond_types: Sequence[BondType] | None, default=[SINGLE, DOUBLE, TRIPLE, AROMATIC]
   :param stereos: the known bond stereochemistries. See [1]_ for more details
   :type stereos: Sequence[int] | None, default=[0, 1, 2, 3, 4, 5]

   .. rubric:: References

   .. [1] https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondStereo.values

   .. py:property:: subfeatures
      :type: list[tuple[str, slice]]


   .. py:method:: __len__()

      the length of a bond feature vector


   .. py:method:: __call__(b: rdkit.Chem.rdchem.Bond) -> numpy.ndarray

      featurize the bond ``b``



