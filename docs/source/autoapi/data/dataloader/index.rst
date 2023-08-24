:py:mod:`data.dataloader`
=========================

.. py:module:: data.dataloader


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.dataloader.MolGraphDataLoader



Functions
~~~~~~~~~

.. autoapisummary::

   data.dataloader.collate_batch



Attributes
~~~~~~~~~~

.. autoapisummary::

   data.dataloader.TrainingBatch
   data.dataloader.MulticomponentTrainingBatch


.. py:data:: TrainingBatch

   

.. py:data:: MulticomponentTrainingBatch

   

.. py:function:: collate_batch(batch: Iterable[chemprop.v2.data.datasets.Datum]) -> TrainingBatch


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


