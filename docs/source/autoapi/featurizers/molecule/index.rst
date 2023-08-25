:py:mod:`featurizers.molecule`
==============================

.. py:module:: featurizers.molecule


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.molecule.MoleculeMolGraphFeaturizer




.. py:class:: MoleculeMolGraphFeaturizer


   Bases: :py:obj:`chemprop.v2.featurizers.mixins.MolGraphFeaturizerMixin`, :py:obj:`chemprop.v2.featurizers.protos.MoleculeMolGraphFeaturizerProto`

   A :class:`MoleculeMolGraphFeaturizer` is the default implementation of a
   :class:`MoleculeMolGraphFeaturizerProto`

   :param atom_featurizer: the featurizer with which to calculate feature representations of the atoms in a given
                           molecule
   :type atom_featurizer: AtomFeaturizerProto, default=AtomFeaturizer()
   :param bond_featurizer: the featurizer with which to calculate feature representations of the bonds in a given
                           molecule
   :type bond_featurizer: BondFeaturizerProto, default=BondFeaturizer()
   :param bond_messages: whether to prepare the `MolGraph`s for use with message passing on bonds
   :type bond_messages: bool, default=True
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



