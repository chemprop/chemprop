:py:mod:`data`
==============

.. py:module:: data


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   dataloader/index.rst
   datapoints/index.rst
   datasets/index.rst
   samplers/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   data.MolGraphDataLoader
   data.MoleculeDatapoint
   data.ReactionDatapoint
   data.MoleculeDataset
   data.ReactionDataset
   data.ClassBalanceSampler
   data.SeededSampler




.. py:class:: MolGraphDataLoader(dataset, batch_size = 50, num_workers = 0, class_balance = False, seed = None, shuffle = True)


   Bases: :py:obj:`torch.utils.data.DataLoader`

   A :class:`MolGraphDataLoader` is a :obj:`~torch.utils.data.DataLoader` for
   :class:`MolGraphDataset`s

   :param dataset: The dataset containing the molecules to load.
   :type dataset: MoleculeDataset
   :param batch_size: the batch size to load
   :type batch_size: int, default=50
   :param num_workers: the number of workers used to build batches.
   :type num_workers: int, default=0
   :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive and negative
                         molecules). Class balance is only available for single task classification datasets. Set
                         shuffle to True in order to get a random subset of the larger class.
   :type class_balance: bool, default=False
   :param seed: the random seed to use for shuffling (only used when `shuffle` is `True`)
   :type seed: int, default=None
   :param shuffle: whether to shuffle the data during sampling
   :type shuffle: bool, default=False


.. py:class:: MoleculeDatapoint


   Bases: :py:obj:`_DatapointMixin`, :py:obj:`_MoleculeDatapointMixin`

   A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets.

   .. py:attribute:: V_f
      :type: numpy.ndarray | None

      a numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
      ``d_vf`` is the number of additional features that will be concatenated to atom-level features
      *before* message passing

   .. py:attribute:: E_f
      :type: numpy.ndarray | None

      A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
      ``d_ef`` is the number of additional features  containing additional features that will be
      concatenated to bond-level features *before* message passing

   .. py:attribute:: V_d
      :type: numpy.ndarray | None

      A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
      ``d_vd`` is the number of additional features that will be concatenated to atom-level features
      *after* message passing

   .. py:method:: __post_init__(mfs)


   .. py:method:: __len__()


   .. py:method:: calc_features(mfs)



.. py:class:: ReactionDatapoint


   Bases: :py:obj:`_DatapointMixin`, :py:obj:`_ReactionDatapointMixin`

   A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets.

   .. py:method:: __post_init__(mfs)


   .. py:method:: __len__()


   .. py:method:: calc_features(mfs)



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



.. py:class:: ClassBalanceSampler(Y, seed = None, shuffle = False)


   Bases: :py:obj:`torch.utils.data.Sampler`

   A :class:`ClassBalanceSampler` samples data from a :class:`MolGraphDataset` such that
   positive and negative classes are equally sampled

   :param dataset: the dataset from which to sample
   :type dataset: MolGraphDataset
   :param seed: the random seed to use for shuffling (only used when `shuffle` is `True`)
   :type seed: int
   :param shuffle: whether to shuffle the data during sampling
   :type shuffle: bool, default=False

   .. py:method:: __iter__()

      an iterator over indices to sample.


   .. py:method:: __len__()

      the number of indices that will be sampled.



.. py:class:: SeededSampler(N, seed)


   Bases: :py:obj:`torch.utils.data.Sampler`

   A :class`SeededSampler` is a class for iterating through a dataset in a randomly seeded
   fashion

   .. py:method:: __iter__()

      an iterator over indices to sample.


   .. py:method:: __len__()

      the number of indices that will be sampled.



