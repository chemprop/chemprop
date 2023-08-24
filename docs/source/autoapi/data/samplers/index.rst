:py:mod:`data.samplers`
=======================

.. py:module:: data.samplers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   data.samplers.SeededSampler
   data.samplers.ClassBalanceSampler




.. py:class:: SeededSampler(N: int, seed: int)


   Bases: :py:obj:`torch.utils.data.Sampler`

   A SeededSampler is a class for iterating through a dataset in a randomly seeded fashion

   .. py:method:: __iter__() -> Iterator[int]

      an iterator over indices to sample.


   .. py:method:: __len__() -> int

      the number of indices that will be sampled.



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



