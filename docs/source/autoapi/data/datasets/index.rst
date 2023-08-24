:py:mod:`data.datasets`
=======================

.. py:module:: data.datasets


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.datasets.Datum
   data.datasets.MolGraphDatasetMixin
   data.datasets.MoleculeDataset
   data.datasets.ReactionDataset




.. py:class:: Datum


   Bases: :py:obj:`NamedTuple`

   .. py:attribute:: mg
      :type: chemprop.v2.featurizers.MolGraph

      

   .. py:attribute:: V_d
      :type: numpy.ndarray | None

      

   .. py:attribute:: x_f
      :type: numpy.ndarray | None

      

   .. py:attribute:: y
      :type: numpy.ndarray | None

      

   .. py:attribute:: weight
      :type: float

      

   .. py:attribute:: lt_mask
      :type: numpy.ndarray | None

      

   .. py:attribute:: gt_mask
      :type: numpy.ndarray | None

      


.. py:class:: MolGraphDatasetMixin


   .. py:property:: Y
      :type: numpy.ndarray


   .. py:property:: X_f
      :type: numpy.ndarray


   .. py:property:: weights
      :type: numpy.ndarray


   .. py:property:: gt_mask
      :type: numpy.ndarray


   .. py:property:: lt_mask
      :type: numpy.ndarray


   .. py:property:: t
      :type: int | None


   .. py:method:: __len__() -> int


   .. py:method:: normalize_targets(scaler: sklearn.preprocessing.StandardScaler | None = None) -> sklearn.preprocessing.StandardScaler

      Normalizes the targets of the dataset using a :obj:`StandardScaler`

      The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
      each task independently. NOTE: This should only be used for regression datasets.

      :returns: a scaler fit to the targets.
      :rtype: StandardScaler


   .. py:method:: normalize_inputs(key: str | None = 'X_f', scaler: sklearn.preprocessing.StandardScaler | None = None) -> sklearn.preprocessing.StandardScaler


   .. py:method:: reset()

      Reset the {atom, bond, molecule} features and targets of each datapoint to its
      initial, unnormalized values.



.. py:class:: MoleculeDataset


   Bases: :py:obj:`torch.utils.data.Dataset`, :py:obj:`MolGraphDatasetMixin`

   A `MolgraphDataset` composed of `MoleculeDatapoint`s

   :param data: the data from which to create a dataset
   :type data: Iterable[MoleculeDatapoint]
   :param featurizer: the featurizer with which to generate MolGraphs of the molecules
   :type featurizer: MoleculeFeaturizer

   .. py:property:: smiles
      :type: list[str]


   .. py:property:: mols
      :type: list[rdkit.Chem.Mol]


   .. py:property:: V_fs
      :type: list[numpy.ndarray]


   .. py:property:: E_fs
      :type: list[numpy.ndarray]


   .. py:property:: V_ds
      :type: list[numpy.ndarray]


   .. py:property:: d_vf
      :type: int | None


   .. py:property:: d_ef
      :type: int | None


   .. py:property:: d_vd
      :type: int | None


   .. py:attribute:: data
      :type: list[chemprop.v2.data.datapoints.MoleculeDatapoint]

      

   .. py:attribute:: featurizer
      :type: chemprop.v2.featurizers.MoleculeMolGraphFeaturizerProto

      

   .. py:method:: __post_init__()


   .. py:method:: __getitem__(idx: int) -> Datum


   .. py:method:: normalize_inputs(key: str | None = 'X_f', scaler: sklearn.preprocessing.StandardScaler | None = None) -> sklearn.preprocessing.StandardScaler


   .. py:method:: reset()

      Reset the {atom, bond, molecule} features and targets of each datapoint to its
      initial, unnormalized values.



.. py:class:: ReactionDataset


   Bases: :py:obj:`torch.utils.data.Dataset`, :py:obj:`MolGraphDatasetMixin`

   A :class:`MolgraphDataset` composed of :class:`ReactionDatapoint`s

   .. py:property:: smiles
      :type: list[str]


   .. py:property:: mols
      :type: list[rdkit.Chem.Mol]


   .. py:attribute:: data
      :type: list[chemprop.v2.data.datapoints.ReactionDatapoint]

      the dataset from which to load

   .. py:attribute:: featurizer
      :type: chemprop.v2.featurizers.ReactionMolGraphFeaturizerProto

      the featurizer with which to generate MolGraphs of the input

   .. py:method:: __getitem__(idx: int) -> Datum



