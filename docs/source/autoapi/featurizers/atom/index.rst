:py:mod:`featurizers.atom`
==========================

.. py:module:: featurizers.atom


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.atom.AtomFeaturizerProto
   featurizers.atom.AtomFeaturizer




.. py:class:: AtomFeaturizerProto


   Bases: :py:obj:`Protocol`

   An :class:`AtomFeaturizerProto` is a protocol for classes that calculate feature vectors of
   RDKit atoms.

   .. py:method:: __len__() -> int

      the length of an atomic feature vector


   .. py:method:: __call__(a: rdkit.Chem.rdchem.Atom) -> numpy.ndarray

      featurize the atom ``a``



.. py:class:: AtomFeaturizer(max_atomic_num: int = 100, degrees: Sequence[int] | None = None, formal_charges: Sequence[int] | None = None, chiral_tags: Sequence[int] | None = None, num_Hs: Sequence[int] | None = None, hybridizations: Sequence[rdkit.Chem.rdchem.HybridizationType] | None = None)


   Bases: :py:obj:`chemprop.v2.featurizers.utils.MultiHotFeaturizerMixin`, :py:obj:`AtomFeaturizerProto`

   An `AtomFeaturizer` calculates feature vectors of RDKit atoms.

   The featurizations produced by this featurizer have the following (general) signature:

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


   **NOTE**: the above signature only applies for the default arguments, as the each slice (save
   for the final two) can increase in size depending on the input arguments.

   :param max_atomic_num: the maximum atomic number categorized, by
   :type max_atomic_num: int, default=100
   :param degrees: the categories for the atomic degree
   :type degrees: Sequence[int] | None, default=[0, 1, 2, 3, 4, 5]
   :param formal_charges: the categories for formal charge of an atom
   :type formal_charges: Sequence[int] | None, default=[-1, -2, 1, 2, 0]
   :param chiral_tags: the categories for the chirality of an atom
   :type chiral_tags: Sequence[int] | None, default=[0, 1, 2, 3]
   :param num_Hs: the categories for the number of hydrogens attached to an atom
   :type num_Hs: Sequence[int] | None, default=[0, 1, 2, 3, 4]
   :param hybridizations: the categories for the hybridization of an atom
   :type hybridizations: Sequence[HybridizationType] | None, default=[SP, SP2, SP3, SP3D, SP3D2]

   .. py:property:: choicess
      :type: list[Sequence]


   .. py:property:: subfeatures
      :type: list[str, slice]


   .. py:method:: __len__() -> int

      the length of an atomic feature vector


   .. py:method:: __call__(a: rdkit.Chem.rdchem.Atom) -> numpy.ndarray

      featurize the atom ``a``


   .. py:method:: num_only(a: rdkit.Chem.rdchem.Atom) -> numpy.ndarray

      featurize the atom by setting only the atomic number bit



