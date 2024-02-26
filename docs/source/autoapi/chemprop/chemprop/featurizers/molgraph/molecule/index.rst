:py:mod:`chemprop.chemprop.featurizers.molgraph.molecule`
=========================================================

.. py:module:: chemprop.chemprop.featurizers.molgraph.molecule


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molgraph.molecule.MoleculeMolGraphFeaturizer
   chemprop.chemprop.featurizers.molgraph.molecule.SimpleMoleculeMolGraphFeaturizer




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



