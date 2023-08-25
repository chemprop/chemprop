:py:mod:`featurizers.reaction`
==============================

.. py:module:: featurizers.reaction


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.reaction.RxnMode
   featurizers.reaction.RxnMolGraphFeaturizer




.. py:class:: RxnMode(*args, **kwds)


   Bases: :py:obj:`chemprop.v2.utils.utils.AutoName`

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


   Bases: :py:obj:`chemprop.v2.featurizers.mixins.MolGraphFeaturizerMixin`, :py:obj:`chemprop.v2.featurizers.protos.RxnMolGraphFeaturizerProto`

   A :class:`ReactionMolGraphFeaturizer` featurizes reactions using the condensed reaction graph method utilized in [1]_

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
      :type: RxnMode


   .. py:attribute:: mode_
      :type: dataclasses.InitVar[str | RxnMode]

      

   .. py:method:: __post_init__(mode_)


   .. py:method:: featurize(rxn, atom_features_extra = None, bond_features_extra = None)


   .. py:method:: map_reac_to_prod(reactants, products)
      :classmethod:

      Map atom indices between corresponding atoms in the reactant and product molecules

      :param reactants: An RDKit molecule of the reactants
      :type reactants: Chem.Mol
      :param products: An RDKit molecule of the products
      :type products: Chem.Mol

      :returns: * **ri2pi** (*dict[int, int]*) -- A dictionary of corresponding atom indices from reactant atoms to product atoms
                * **pdt_idxs** (*list[int]*) -- atom indices of poduct atoms
                * **rct_idxs** (*list[int]*) -- atom indices of reactant atoms



