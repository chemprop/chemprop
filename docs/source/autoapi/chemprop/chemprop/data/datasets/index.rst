:py:mod:`chemprop.chemprop.data.datasets`
=========================================

.. py:module:: chemprop.chemprop.data.datasets


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.datasets.Datum
   chemprop.chemprop.data.datasets.MolGraphDataset
   chemprop.chemprop.data.datasets.MoleculeDataset
   chemprop.chemprop.data.datasets.ReactionDataset
   chemprop.chemprop.data.datasets.MulticomponentDataset




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

      


.. py:class:: MolGraphDataset


   Bases: :py:obj:`torch.utils.data.Dataset`

   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.

   .. py:method:: __getitem__(idx)
      :abstractmethod:



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



