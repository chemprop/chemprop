:py:mod:`data.datasets`
=======================

.. py:module:: data.datasets


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.datasets.Datum
   data.datasets.MoleculeDataset
   data.datasets.ReactionDataset




.. py:class:: Datum


   Bases: :py:obj:`NamedTuple`

   a singular training data point

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

      


.. py:class:: MoleculeDataset


   Bases: :py:obj:`torch.utils.data.Dataset`, :py:obj:`_MolGraphDatasetMixin`

   A `MolgraphDataset` composed of `MoleculeDatapoint`s

   :param data: the data from which to create a dataset
   :type data: Iterable[MoleculeDatapoint]
   :param featurizer: the featurizer with which to generate MolGraphs of the molecules
   :type featurizer: MoleculeFeaturizer

   .. py:property:: smiles
      :type: list[str]

      the SMILES strings associated with the dataset

   .. py:property:: mols
      :type: list[rdkit.Chem.Mol]

      the molecules associated with the dataset

   .. py:property:: V_fs
      :type: list[numpy.ndarray]

      the (scaled) atom descriptors of the dataset

   .. py:property:: E_fs
      :type: list[numpy.ndarray]

      the (scaled) bond features of the dataset

   .. py:property:: V_ds
      :type: list[numpy.ndarray]

      the (scaled) atom descriptors of the dataset

   .. py:property:: d_vf
      :type: int | None

      the extra atom feature dimension, if any

   .. py:property:: d_ef
      :type: int | None

      the extra bond feature dimension, if any

   .. py:property:: d_vd
      :type: int | None

      the extra atom descriptor dimension, if any

   .. py:attribute:: data
      :type: list[chemprop.v2.data.datapoints.MoleculeDatapoint]

      

   .. py:attribute:: featurizer
      :type: chemprop.v2.featurizers.MoleculeMolGraphFeaturizerProto

      

   .. py:method:: __post_init__()


   .. py:method:: __getitem__(idx)


   .. py:method:: normalize_inputs(key = 'X_f', scaler = None)


   .. py:method:: reset()

      reset the {atom, bond, molecule} features and targets of each datapoint to its raw
      value



.. py:class:: ReactionDataset


   Bases: :py:obj:`torch.utils.data.Dataset`, :py:obj:`_MolGraphDatasetMixin`

   A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`s

   .. py:property:: smiles
      :type: list[str]


   .. py:property:: mols
      :type: list[rdkit.Chem.Mol]


   .. py:attribute:: data
      :type: list[chemprop.v2.data.datapoints.ReactionDatapoint]

      the dataset from which to load

   .. py:attribute:: featurizer
      :type: chemprop.v2.featurizers.RxnMolGraphFeaturizerProto

      the featurizer with which to generate MolGraphs of the input

   .. py:method:: __getitem__(idx)



