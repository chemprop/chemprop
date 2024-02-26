:py:mod:`chemprop.chemprop.featurizers.molgraph.reaction`
=========================================================

.. py:module:: chemprop.chemprop.featurizers.molgraph.reaction


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molgraph.reaction.RxnMode
   chemprop.chemprop.featurizers.molgraph.reaction.RxnMolGraphFeaturizer
   chemprop.chemprop.featurizers.molgraph.reaction.CondensedGraphOfReactionFeaturizer




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molgraph.reaction.CGRFeaturizer


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

   

