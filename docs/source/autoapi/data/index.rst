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
   data.MolGraphDatasetMixin
   data.MoleculeDataset
   data.ReactionDataset
   data.ClassBalanceSampler
   data.SeededSampler




.. py:class:: MolGraphDataLoader(dataset: chemprop.v2.data.datasets.MolGraphDatasetMixin, batch_size: int = 50, num_workers: int = 0, class_balance: bool = False, seed: int | None = None, shuffle: bool = True)


   Bases: :py:obj:`torch.utils.data.DataLoader`

   A `MolGraphDataLoader` is a DataLoader for `MolGraphDataset`s

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



.. py:class:: ReactionDatapoint


   Bases: :py:obj:`DatapointMixin`, :py:obj:`ReactionDatapointMixin`

   A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets.

   .. py:method:: __post_init__(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto] | None)


   .. py:method:: __len__() -> int


   .. py:method:: calc_features(mfs: list[chemprop.v2.featurizers.featurizers.MoleculeFeaturizerProto]) -> numpy.ndarray



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



.. py:class:: ClassBalanceSampler(Y: numpy.ndarray, seed: Optional[int] = None, shuffle: bool = False)


   Bases: :py:obj:`torch.utils.data.Sampler`

   A `ClassBalanceSampler` samples data from a `MolGraphDataset` such that positive and
   negative classes are equally sampled

   :param dataset: the dataset from which to sample
   :type dataset: MolGraphDataset
   :param seed: the random seed to use for shuffling (only used when `shuffle` is `True`)
   :type seed: int
   :param shuffle: whether to shuffle the data during sampling
   :type shuffle: bool, default=False

   .. py:method:: __iter__() -> Iterator[int]

      an iterator over indices to sample.


   .. py:method:: __len__() -> int

      the number of indices that will be sampled.



.. py:class:: SeededSampler(N: int, seed: int)


   Bases: :py:obj:`torch.utils.data.Sampler`

   A SeededSampler is a class for iterating through a dataset in a randomly seeded fashion

   .. py:method:: __iter__() -> Iterator[int]

      an iterator over indices to sample.


   .. py:method:: __len__() -> int

      the number of indices that will be sampled.



