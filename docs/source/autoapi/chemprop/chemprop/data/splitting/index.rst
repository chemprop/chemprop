:py:mod:`chemprop.chemprop.data.splitting`
==========================================

.. py:module:: chemprop.chemprop.data.splitting


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.splitting.SplitType



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.splitting.split_data
   chemprop.chemprop.data.splitting.split_monocomponent
   chemprop.chemprop.data.splitting.split_multicomponent



Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.splitting.logger
   chemprop.chemprop.data.splitting.MulticomponentDatapoint


.. py:data:: logger

   

.. py:data:: MulticomponentDatapoint

   

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

      


.. py:function:: split_data(datapoints, split = 'random', sizes = (0.8, 0.1, 0.1), seed = 0, num_folds = 1)

   Splits data into training, validation, and test splits.

   :param datapoints: Sequence of chemprop.data.MoleculeDatapoint.
   :type datapoints: Sequence[MoleculeDatapoint]
   :param split: Split type, one of ~chemprop.data.utils.SplitType, by default "random"
   :type split: SplitType | str, optional
   :param sizes: 3-tuple with the proportions of data in the train, validation, and test sets, by default (0.8, 0.1, 0.1)
   :type sizes: tuple[float, float, float], optional
   :param seed: The random seed passed to astartes, by default 0
   :type seed: int, optional
   :param num_folds: Number of folds to create (only needed for "cv" and "cv-no-test"), by default 1
   :type num_folds: int, optional

   :returns: A tuple of list of indices corresponding to the train, validation, and test splits of the data.
             If the split type is "cv" or "cv-no-test", returns a tuple of lists of lists of indices corresponding to the train, validation, and test splits of each fold.
                 NOTE: validation may or may not be present
   :rtype: tuple[list[int], list[int], list[int]] | tuple[list[list[int], ...], list[list[int], ...], list[list[int], ...]]

   :raises ValueError: Requested split sizes tuple not of length 3
   :raises ValueError: Innapropriate number of folds requested
   :raises ValueError: Unsupported split method requested


.. py:function:: split_monocomponent(datapoints, split = 'random', **kwargs)

   Splits monocomponent data into training, validation, and test splits.


.. py:function:: split_multicomponent(datapointss, split = 'random', key_index = 0, **kwargs)

   Splits multicomponent data into training, validation, and test splits.


