:py:mod:`featurizers.reaction`
==============================

.. py:module:: featurizers.reaction


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.reaction.ReactionMolGraphFeaturizerProto
   featurizers.reaction.ReactionMolGraphFeaturizer




.. py:class:: ReactionMolGraphFeaturizerProto


   Bases: :py:obj:`chemprop.v2.featurizers.proto.MolGraphFeaturizerProto`

   A :class:`ReactionMolGraphFeaturizerProto` featurizes reactions (i.e., a 2-tuple of reactant
   and product molecules) into :class:`MolGraph`s

   .. py:method:: __call__(reaction: tuple[rdkit.Chem.Mol, rdkit.Chem.Mol], atom_features_extra: numpy.ndarray | None = None, bond_features_extra: numpy.ndarray | None = None) -> chemprop.v2.featurizers.molgraph.MolGraph

      Featurize the input reaction into a molecular graph

      :param reaction: a 2-tuple of atom-mapped rdkit molecules, where the 0th element is the reactant and the
                       1st element is the product
      :type reaction: tuple[Chem.Mol, Chem.Mol]
      :param atom_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type atom_features_extra: np.ndarray | None, default=None
      :param bond_features_extra: *UNSUPPORTED* maintained only to maintain parity with the method signature of the
                                  `MoleculeFeaturizer`
      :type bond_features_extra: np.ndarray | None, default=None

      :returns: the molecular graph of the reaction
      :rtype: MolGraph



.. py:class:: ReactionMolGraphFeaturizer


   Bases: :py:obj:`chemprop.v2.featurizers.mixins.MolGraphFeaturizerMixin`, :py:obj:`ReactionMolGraphFeaturizerProto`

   Featurize reactions using the condensed reaction graph method utilized in [1]_

   **NOTE**: This class *does not* accept a :class:`AtomFeaturizerProto` instance. This is because
   it requries the :meth:`num_only()` method, which is only implemented in the concrete
   :class:`AtomFeaturizer` class

   :param atom_featurizer: the featurizer with which to calculate feature representations of the atoms in a given
                           molecule
   :type atom_featurizer: AtomFeaturizer, default=AtomFeaturizer()
   :param bond_featurizer: the featurizer with which to calculate feature representations of the bonds in a given
                           molecule
   :type bond_featurizer: BondFeaturizerBase, default=BondFeaturizer()
   :param bond_messages: whether to prepare the `MolGraph`s for use with bond-based message-passing
   :type bond_messages: bool, default=True
   :param mode: the mode by which to featurize the reaction as either the string code or enum value
   :type mode: Union[str, ReactionMode], default=ReactionMode.REAC_DIFF

   .. rubric:: References

   .. [1] Heid, E.; Green, W.H. "Machine Learning of Reaction Properties via Learned
       Representations of the Condensed Graph of Reaction." J. Chem. Inf. Model. 2022, 62,
       2101-2110. https://doi.org/10.1021/acs.jcim.1c00975

   .. py:property:: mode
      :type: chemprop.v2.featurizers.utils.ReactionMode


   .. py:attribute:: mode_
      :type: dataclasses.InitVar[str | chemprop.v2.featurizers.utils.ReactionMode]

      

   .. py:method:: __post_init__(mode_: str | chemprop.v2.featurizers.utils.ReactionMode)


   .. py:method:: featurize(reaction: tuple[rdkit.Chem.Mol, rdkit.Chem.Mol], atom_features_extra: numpy.ndarray | None = None, bond_features_extra: numpy.ndarray | None = None) -> chemprop.v2.featurizers.molgraph.MolGraph


   .. py:method:: map_reac_to_prod(reactants: rdkit.Chem.Mol, products: rdkit.Chem.Mol) -> tuple[dict[int, int], list[int], list[int]]
      :staticmethod:

      Map atom indices between corresponding atoms in the reactant and product molecules

      :param reactants: An RDKit molecule of the reactants
      :type reactants: Chem.Mol
      :param products: An RDKit molecule of the products
      :type products: Chem.Mol

      :returns: * **ri2pi** (*dict[int, int]*) -- A dictionary of corresponding atom indices from reactant atoms to product atoms
                * **pdt_idxs** (*list[int]*) -- atom indices of poduct atoms
                * **rct_idxs** (*list[int]*) -- atom indices of reactant atoms



