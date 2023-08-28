:py:mod:`featurizers.bond`
==========================

.. py:module:: featurizers.bond


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.bond.BondFeaturizer




.. py:class:: BondFeaturizer(bond_types = None, stereos = None)


   Bases: :py:obj:`chemprop.v2.featurizers.protos.BondFeaturizerProto`

   A :class:`BondFeaturizer` feauturizes bonds based on the following attributes:

   * ``null``-ity (i.e., is the bond ``None``?)
   * bond type
   * conjugated?
   * in ring?
   * stereochemistry

   The feature vectors produced by this featurizer have the following (general) signature:

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

   .. py:method:: __len__()

      the length of a bond feature vector


   .. py:method:: __call__(b)

      featurize the bond ``b``


   .. py:method:: one_hot_index(x, xs)
      :classmethod:

      the index of ``x`` in ``xs``, if it exists. Otherwise, return ``len(xs) + 1``.



