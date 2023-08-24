:py:mod:`data.datapoints`
=========================

.. py:module:: data.datapoints


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.datapoints.DatapointMixin
   data.datapoints.MoleculeDatapointMixin
   data.datapoints.MoleculeDatapoint
   data.datapoints.ReactionDatapointMixin
   data.datapoints.ReactionDatapoint
   data.datapoints.MulticomponentDatapointMixin
   data.datapoints.MulticomponentDatapoint




.. py:class:: DatapointMixin


   A mixin class for both molecule- and reaction- and multicomponent-type data

   .. py:property:: t
      :type: int | None


   .. py:attribute:: y
      :type: numpy.ndarray | None

      the targets for the molecule with unknown targets indicated by `nan`s

   .. py:attribute:: weight
      :type: float
      :value: 1

      the weight of this datapoint for the loss calculation.

   .. py:attribute:: gt_mask
      :type: numpy.ndarray | None

      Indicates whether the targets are an inequality regression target of the form `<x`

   .. py:attribute:: lt_mask
      :type: numpy.ndarray | None

      Indicates whether the targets are an inequality regression target of the form `>x`

   .. py:attribute:: x_f
      :type: numpy.ndarray | None

      A vector of length ``d_f`` containing additional features (e.g., Morgan fingerprint) that
      will be concatenated to the global representation *after* aggregation

   .. py:attribute:: mfs
      :type: dataclasses.InitVar[list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None]

      A list of molecule featurizers to use

   .. py:attribute:: x_phase
      :type: list[float]

      A one-hot vector indicating the phase of the data, as used in spectra data.

   .. py:method:: __post_init__(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None)



.. py:class:: MoleculeDatapointMixin


   .. py:attribute:: mol
      :type: rdkit.Chem.AllChem.Mol

      the molecule associated with this datapoint

   .. py:method:: from_smi(smi: str, *args, keep_h: bool = False, add_h: bool = False, **kwargs) -> MoleculeDatapointMixin
      :classmethod:



.. py:class:: MoleculeDatapoint


   Bases: :py:obj:`DatapointMixin`, :py:obj:`MoleculeDatapointMixin`

   A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets.

   .. py:attribute:: V_f
      :type: numpy.ndarray | None

      a numpy array of shape `V x d_vf`, where `V` is the number of atoms in the molecule, and
      `d_vf` is the number of additional features that will be concatenated to atom-level features
      _before_ message passing

   .. py:attribute:: E_f
      :type: numpy.ndarray | None

      A numpy array of shape `E x d_ef`, where `E` is the number of bonds in the molecule, and
      `d_ef` is the number of additional features  containing additional features that will be
      concatenated to bond-level features _before_ message passing

   .. py:attribute:: V_d
      :type: numpy.ndarray | None

      A numpy array of shape `V x d_vd`, where `V` is the number of atoms in the molecule, and
      `d_vd` is the number of additional features that will be concatenated to atom-level features
      _after_ message passing

   .. py:method:: __post_init__(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None)


   .. py:method:: __len__() -> int


   .. py:method:: calc_features(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto]) -> numpy.ndarray



.. py:class:: ReactionDatapointMixin


   .. py:attribute:: rct
      :type: rdkit.Chem.AllChem.Mol

      the reactant associated with this datapoint

   .. py:attribute:: pdt
      :type: rdkit.Chem.AllChem.Mol

      the product associated with this datapoint

   .. py:method:: from_smi(rxn_or_smis: str | tuple[str, str], keep_h: bool = False, add_h: bool = False, *args, **kwargs) -> ReactionDatapointMixin
      :classmethod:



.. py:class:: ReactionDatapoint


   Bases: :py:obj:`DatapointMixin`, :py:obj:`ReactionDatapointMixin`

   A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets.

   .. py:method:: __post_init__(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None)


   .. py:method:: __len__() -> int


   .. py:method:: calc_features(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto]) -> numpy.ndarray



.. py:class:: MulticomponentDatapointMixin


   .. py:attribute:: mols
      :type: list[rdkit.Chem.AllChem.Mol]

      the molecules associated with this datapoint

   .. py:method:: from_smis(smis: list[str], keep_h: bool = False, add_h: bool = False, *args, **kwargs) -> MulticomponentDatapointMixin
      :classmethod:



.. py:class:: MulticomponentDatapoint


   Bases: :py:obj:`DatapointMixin`, :py:obj:`MulticomponentDatapointMixin`

   A :class:`MulticomponentDatapoint` contains a list of molecules and their associated features and targets.

   .. py:method:: __post_init__(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None)


   .. py:method:: __len__() -> int


   .. py:method:: calc_features(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto]) -> numpy.ndarray



