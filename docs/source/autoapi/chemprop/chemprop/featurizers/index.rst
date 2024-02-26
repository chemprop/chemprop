:py:mod:`chemprop.chemprop.featurizers`
=======================================

.. py:module:: chemprop.chemprop.featurizers


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   molgraph/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   atom/index.rst
   bond/index.rst
   molecule/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.MultiHotAtomFeaturizer
   chemprop.chemprop.featurizers.AtomFeaturizer
   chemprop.chemprop.featurizers.MultiHotBondFeaturizer
   chemprop.chemprop.featurizers.BondFeaturizer
   chemprop.chemprop.featurizers.MolGraph
   chemprop.chemprop.featurizers.MoleculeMolGraphFeaturizer
   chemprop.chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer
   chemprop.chemprop.featurizers.RxnMolGraphFeaturizer
   chemprop.chemprop.featurizers.CondensedGraphOfReactionFeaturizer
   chemprop.chemprop.featurizers.RxnMode
   chemprop.chemprop.featurizers.MoleculeFeaturizer
   chemprop.chemprop.featurizers.MorganFeaturizerMixin
   chemprop.chemprop.featurizers.BinaryFeaturizerMixin
   chemprop.chemprop.featurizers.CountFeaturizerMixin
   chemprop.chemprop.featurizers.MorganBinaryFeaturzer
   chemprop.chemprop.featurizers.MorganCountFeaturizer




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.CGRFeaturizer
   chemprop.chemprop.featurizers.MoleculeFeaturizerRegistry


.. py:class:: MultiHotAtomFeaturizer


   Bases: :py:obj:`AtomFeaturizer`

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



.. py:class:: AtomFeaturizer


   Bases: :py:obj:`abc.ABC`

   An :class:`AtomFeaturizer` calculates feature vectors of RDKit atoms.

   .. py:method:: __len__()
      :abstractmethod:

      the length of an atomic feature vector


   .. py:method:: __call__(a)
      :abstractmethod:

      featurize the atom ``a``



.. py:class:: MultiHotBondFeaturizer(bond_types = None, stereos = None)


   Bases: :py:obj:`BondFeaturizer`

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



.. py:class:: BondFeaturizer


   Bases: :py:obj:`abc.ABC`

   A :class:`BondFeaturizer` calculates feature vectors of RDKit bonds

   .. py:method:: __len__()
      :abstractmethod:

      the length of a bond feature vector


   .. py:method:: __call__(b)
      :abstractmethod:

      featurize the bond ``b``



.. py:class:: MolGraph


   Bases: :py:obj:`NamedTuple`

   A :class:`MolGraph` represents the graph featurization of a molecule.

   .. py:attribute:: V
      :type: numpy.ndarray

      an array of shape ``V x d_v`` containing the atom features of the molecule

   .. py:attribute:: E
      :type: numpy.ndarray

      an array of shape ``E x d_e`` containing the bond features of the molecule

   .. py:attribute:: edge_index
      :type: numpy.ndarray

      an array of shape ``2 x E`` containing the edges of the graph in COO format

   .. py:attribute:: rev_edge_index
      :type: numpy.ndarray

      A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute.


.. py:class:: MoleculeMolGraphFeaturizer


   Bases: :py:obj:`abc.ABC`

   A :class:`MoleculeMolGraphFeaturizer` featurizes RDKit molecules into
   :class:`MolGraph`\s

   .. py:method:: __call__(mol, atom_features_extra = None, bond_features_extra = None)
      :abstractmethod:

      Featurize the input molecule into a molecular graph

      :param mol: the input molecule
      :type mol: Chem.Mol
      :param atom_features_extra: Additional features to concatenate to the calculated atom features
      :type atom_features_extra: np.ndarray | None, default=None
      :param bond_features_extra: Additional features to concatenate to the calculated bond features
      :type bond_features_extra: np.ndarray | None, default=None

      :returns: the molecular graph of the molecule
      :rtype: MolGraph



.. py:class:: SimpleMoleculeMolGraphFeaturizer


   Bases: :py:obj:`chemprop.featurizers.molgraph.mixins._MolGraphFeaturizerMixin`, :py:obj:`MoleculeMolGraphFeaturizer`

   A :class:`SimpleMoleculeMolGraphFeaturizer` is the default implementation of a
   :class:`MoleculeMolGraphFeaturizer`

   :param atom_featurizer: the featurizer with which to calculate feature representations of the atoms in a given
                           molecule
   :type atom_featurizer: AtomFeaturizer, default=MultiHotAtomFeaturizer()
   :param bond_featurizer: the featurizer with which to calculate feature representations of the bonds in a given
                           molecule
   :type bond_featurizer: BondFeaturizer, default=MultiHotBondFeaturizer()
   :param extra_atom_fdim: the dimension of the additional features that will be concatenated onto the calculated
                           features of each atom
   :type extra_atom_fdim: int, default=0
   :param extra_bond_fdim: the dimension of the additional features that will be concatenated onto the calculated
                           features of each bond
   :type extra_bond_fdim: int, default=0

   .. py:attribute:: extra_atom_fdim
      :type: dataclasses.InitVar[int]
      :value: 0

      

   .. py:attribute:: extra_bond_fdim
      :type: dataclasses.InitVar[int]
      :value: 0

      

   .. py:method:: __post_init__(extra_atom_fdim = 0, extra_bond_fdim = 0)


   .. py:method:: __call__(mol, atom_features_extra = None, bond_features_extra = None)

      Featurize the input molecule into a molecular graph

      :param mol: the input molecule
      :type mol: Chem.Mol
      :param atom_features_extra: Additional features to concatenate to the calculated atom features
      :type atom_features_extra: np.ndarray | None, default=None
      :param bond_features_extra: Additional features to concatenate to the calculated bond features
      :type bond_features_extra: np.ndarray | None, default=None

      :returns: the molecular graph of the molecule
      :rtype: MolGraph



.. py:class:: RxnMolGraphFeaturizer


   Bases: :py:obj:`abc.ABC`

   A :class:`RxnMolGraphFeaturizer` featurizes reactions (i.e., a 2-tuple of reactant
   and product molecules) into :class:`MolGraph`\s

   .. py:method:: __call__(rxn, atom_features_extra = None, bond_features_extra = None)
      :abstractmethod:

      Featurize the input reaction into a molecular graph

      :param rxn: a 2-tuple of atom-mapped rdkit molecules, where the 0th element is the reactant and the
                  1st element is the product
      :type rxn: tuple[Chem.Mol, Chem.Mol]
      :param atom_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type atom_features_extra: np.ndarray | None, default=None
      :param bond_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type bond_features_extra: np.ndarray | None, default=None

      :returns: the molecular graph of the reaction
      :rtype: MolGraph



.. py:class:: CondensedGraphOfReactionFeaturizer


   Bases: :py:obj:`chemprop.featurizers.molgraph.mixins._MolGraphFeaturizerMixin`, :py:obj:`RxnMolGraphFeaturizer`

   A :class:`CondensedGraphOfReactionFeaturizer` featurizes reactions using the condensed reaction graph method utilized in [1]_

   **NOTE**: This class *does not* accept a :class:`AtomFeaturizer` instance. This is because
   it requries the :meth:`num_only()` method, which is only implemented in the concrete
   :class:`AtomFeaturizer` class

   :param atom_featurizer: the featurizer with which to calculate feature representations of the atoms in a given
                           molecule
   :type atom_featurizer: AtomFeaturizer, default=AtomFeaturizer()
   :param bond_featurizer: the featurizer with which to calculate feature representations of the bonds in a given
                           molecule
   :type bond_featurizer: BondFeaturizerBase, default=BondFeaturizer()
   :param mode_: the mode by which to featurize the reaction as either the string code or enum value
   :type mode_: Union[str, ReactionMode], default=ReactionMode.REAC_DIFF

   .. rubric:: References

   .. [1] Heid, E.; Green, W.H. "Machine Learning of Reaction Properties via Learned
       Representations of the Condensed Graph of Reaction." J. Chem. Inf. Model. 2022, 62,
       2101-2110. https://doi.org/10.1021/acs.jcim.1c00975

   .. py:property:: mode
      :type: RxnMode


   .. py:attribute:: mode_
      :type: dataclasses.InitVar[str | RxnMode]

      

   .. py:method:: __post_init__(mode_)


   .. py:method:: __call__(rxn, atom_features_extra = None, bond_features_extra = None)

      Featurize the input reaction into a molecular graph

      :param rxn: a 2-tuple of atom-mapped rdkit molecules, where the 0th element is the reactant and the
                  1st element is the product
      :type rxn: tuple[Chem.Mol, Chem.Mol]
      :param atom_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type atom_features_extra: np.ndarray | None, default=None
      :param bond_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type bond_features_extra: np.ndarray | None, default=None

      :returns: the molecular graph of the reaction
      :rtype: MolGraph


   .. py:method:: map_reac_to_prod(reacs, pdts)
      :classmethod:

      Map atom indices between corresponding atoms in the reactant and product molecules

      :param reacs: An RDKit molecule of the reactants
      :type reacs: Chem.Mol
      :param pdts: An RDKit molecule of the products
      :type pdts: Chem.Mol

      :returns: * **ri2pi** (*dict[int, int]*) -- A dictionary of corresponding atom indices from reactant atoms to product atoms
                * **pdt_idxs** (*list[int]*) -- atom indices of poduct atoms
                * **rct_idxs** (*list[int]*) -- atom indices of reactant atoms



.. py:data:: CGRFeaturizer

   

.. py:class:: RxnMode


   Bases: :py:obj:`chemprop.utils.utils.EnumMapping`

   The mode by which a reaction should be featurized into a `MolGraph`

   .. py:attribute:: REAC_PROD

      concatenate the reactant features with the product features.

   .. py:attribute:: REAC_PROD_BALANCE

      concatenate the reactant features with the products feature and balances imbalanced
      reactions

   .. py:attribute:: REAC_DIFF

      concatenates the reactant features with the difference in features between reactants and
      products

   .. py:attribute:: REAC_DIFF_BALANCE

      concatenates the reactant features with the difference in features between reactants and
      product and balances imbalanced reactions

   .. py:attribute:: PROD_DIFF

      concatenates the product features with the difference in features between reactants and
      products

   .. py:attribute:: PROD_DIFF_BALANCE

      concatenates the product features with the difference in features between reactants and
      products and balances imbalanced reactions


.. py:class:: MoleculeFeaturizer


   Bases: :py:obj:`abc.ABC`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.

   .. py:method:: __len__()
      :abstractmethod:

      the length of the feature vector


   .. py:method:: __call__(mol)
      :abstractmethod:

      Featurize the molecule ``mol``



.. py:data:: MoleculeFeaturizerRegistry

   

.. py:class:: MorganFeaturizerMixin(radius = 2, length = 2048, include_chirality = True)


   .. py:method:: __len__()



.. py:class:: BinaryFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: CountFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: MorganBinaryFeaturzer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`BinaryFeaturizerMixin`, :py:obj:`MoleculeFeaturizer`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.


.. py:class:: MorganCountFeaturizer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`CountFeaturizerMixin`, :py:obj:`MoleculeFeaturizer`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.


