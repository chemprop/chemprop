:py:mod:`chemprop.chemprop.data`
================================

.. py:module:: chemprop.chemprop.data


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   collate/index.rst
   dataloader/index.rst
   datapoints/index.rst
   datasets/index.rst
   samplers/index.rst
   splitting/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.BatchMolGraph
   chemprop.chemprop.data.TrainingBatch
   chemprop.chemprop.data.MolGraphDataLoader
   chemprop.chemprop.data.MoleculeDatapoint
   chemprop.chemprop.data.ReactionDatapoint
   chemprop.chemprop.data.MoleculeDataset
   chemprop.chemprop.data.ReactionDataset
   chemprop.chemprop.data.Datum
   chemprop.chemprop.data.MulticomponentDataset
   chemprop.chemprop.data.ClassBalanceSampler
   chemprop.chemprop.data.SeededSampler
   chemprop.chemprop.data.SplitType



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.collate_batch
   chemprop.chemprop.data.collate_multicomponent
   chemprop.chemprop.data.split_monocomponent
   chemprop.chemprop.data.split_multicomponent



.. py:class:: BatchMolGraph


   A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`\s.

   It has all the attributes of a ``MolGraph`` with the addition of the ``batch`` attribute. This
   class is intended for use with data loading, so it uses :obj:`~torch.Tensor`\s to store data

   .. py:attribute:: mgs
      :type: dataclasses.InitVar[Sequence[chemprop.featurizers.MolGraph]]

      A list of individual :class:`MolGraph`s to be batched together

   .. py:attribute:: V
      :type: torch.Tensor

      the atom feature matrix

   .. py:attribute:: E
      :type: torch.Tensor

      the bond feature matrix

   .. py:attribute:: edge_index
      :type: torch.Tensor

      an tensor of shape ``2 x E`` containing the edges of the graph in COO format

   .. py:attribute:: rev_edge_index
      :type: torch.Tensor

      A tensor of shape ``E`` that maps from an edge index to the index of the source of the
      reverse edge in the ``edge_index`` attribute.

   .. py:attribute:: batch
      :type: torch.Tensor

      the index of the parent :class:`MolGraph` in the batched graph

   .. py:method:: __post_init__(mgs)


   .. py:method:: __len__()

      the number of individual :class:`MolGraph`\s in this batch


   .. py:method:: to(device)



.. py:class:: TrainingBatch


   Bases: :py:obj:`NamedTuple`

   .. py:attribute:: bmg
      :type: BatchMolGraph

      

   .. py:attribute:: V_d
      :type: torch.Tensor | None

      

   .. py:attribute:: X_f
      :type: torch.Tensor | None

      

   .. py:attribute:: Y
      :type: torch.Tensor | None

      

   .. py:attribute:: w
      :type: torch.Tensor

      

   .. py:attribute:: lt_mask
      :type: torch.Tensor | None

      

   .. py:attribute:: gt_mask
      :type: torch.Tensor | None

      


.. py:function:: collate_batch(batch)


.. py:function:: collate_multicomponent(batches)


.. py:class:: MolGraphDataLoader(dataset, batch_size = 50, num_workers = 0, class_balance = False, seed = None, shuffle = True, **kwargs)


   Bases: :py:obj:`torch.utils.data.DataLoader`

   A :class:`MolGraphDataLoader` is a :obj:`~torch.utils.data.DataLoader` for
   :class:`MolGraphDataset`\s

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


   Bases: :py:obj:`_MolGraphDatasetMixin`, :py:obj:`MolGraphDataset`

   A :class:`MolgraphDataset` composed of :class:`MoleculeDatapoint`\s

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
      :type: numpy.ndarray

      the (scaled) atom descriptors of the dataset

   .. py:property:: E_fs
      :type: numpy.ndarray

      the (scaled) bond features of the dataset

   .. py:property:: V_ds
      :type: numpy.ndarray

      the (scaled) atom descriptors of the dataset

   .. py:property:: d_vf
      :type: int

      the extra atom feature dimension, if any

   .. py:property:: d_ef
      :type: int

      the extra bond feature dimension, if any

   .. py:property:: d_vd
      :type: int

      the extra atom descriptor dimension, if any

   .. py:attribute:: data
      :type: list[chemprop.data.datapoints.MoleculeDatapoint]

      

   .. py:attribute:: featurizer
      :type: chemprop.featurizers.MoleculeMolGraphFeaturizer

      

   .. py:method:: __post_init__()


   .. py:method:: __getitem__(idx)


   .. py:method:: normalize_inputs(key = 'X_f', scaler = None)


   .. py:method:: reset()

      reset the {atom, bond, molecule} features and targets of each datapoint to its raw
      value



.. py:class:: ReactionDataset


   Bases: :py:obj:`_MolGraphDatasetMixin`, :py:obj:`MolGraphDataset`

   A :class:`ReactionDataset` composed of :class:`ReactionDatapoint`\s

   .. py:property:: smiles
      :type: list[str]


   .. py:property:: mols
      :type: list[rdkit.Chem.Mol]


   .. py:attribute:: data
      :type: list[chemprop.data.datapoints.ReactionDatapoint]

      the dataset from which to load

   .. py:attribute:: featurizer
      :type: chemprop.featurizers.RxnMolGraphFeaturizer

      the featurizer with which to generate MolGraphs of the input

   .. py:method:: __post_init__()


   .. py:method:: __getitem__(idx)



.. py:class:: Datum


   Bases: :py:obj:`NamedTuple`

   a singular training data point

   .. py:attribute:: mg
      :type: chemprop.featurizers.MolGraph

      

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

      


.. py:class:: MulticomponentDataset


   Bases: :py:obj:`_MolGraphDatasetMixin`, :py:obj:`torch.utils.data.Dataset`

   A :class:`MulticomponentDataset` is a :class:`Dataset` composed of parallel :class:`MoleculeDatasets` and :class:`ReactionDataset`\s

   .. py:property:: smiles
      :type: list[list[str]]


   .. py:property:: mols
      :type: list[list[rdkit.Chem.Mol]]


   .. py:attribute:: datasets
      :type: list[MoleculeDataset | ReactionDataset]

      the parallel datasets

   .. py:method:: __post_init__()


   .. py:method:: __len__()


   .. py:method:: __getitem__(idx)


   .. py:method:: normalize_targets(scaler = None)

      Normalizes the targets of this dataset using a :obj:`StandardScaler`

      The :obj:`StandardScaler` subtracts the mean and divides by the standard deviation for
      each task independently. NOTE: This should only be used for regression datasets.

      :returns: a scaler fit to the targets.
      :rtype: StandardScaler


   .. py:method:: normalize_inputs(key = 'X_f', scaler = None)


   .. py:method:: reset()

      Reset the {atom, bond, molecule} features and targets of each datapoint to its
      initial, unnormalized values.



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



.. py:function:: split_monocomponent(datapoints, split = 'random', **kwargs)

   Splits monocomponent data into training, validation, and test splits.


.. py:function:: split_multicomponent(datapointss, split = 'random', key_index = 0, **kwargs)

   Splits multicomponent data into training, validation, and test splits.


.. py:class:: SplitType


   Bases: :py:obj:`chemprop.utils.utils.EnumMapping`

   Enum where members are also (and must be) strings

   .. py:attribute:: CV_NO_VAL

      

   .. py:attribute:: CV

      

   .. py:attribute:: SCAFFOLD_BALANCED

      

   .. py:attribute:: RANDOM_WITH_REPEATED_SMILES

      

   .. py:attribute:: RANDOM

      

   .. py:attribute:: KENNARD_STONE

      

   .. py:attribute:: KMEANS

      


