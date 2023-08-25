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

   

.. py:function:: collate_batch(batch)


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


