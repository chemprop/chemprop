:py:mod:`models.models.multiclass`
==================================

.. py:module:: models.models.multiclass


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.models.multiclass.MulticlassMPNN
   models.models.multiclass.DirichletMulticlassMPNN




.. py:class:: MulticlassMPNN(mp_block, n_tasks, n_classes, *args, **kwargs)


   Bases: :py:obj:`chemprop.v2.models.models.base.MPNN`

   .. py:method:: forward(*args, **kwargs)


   .. py:method:: predict_step(batch, batch_idx, dataloader_idx = 0)



.. py:class:: DirichletMulticlassMPNN(*args, **kwargs)


   Bases: :py:obj:`MulticlassMPNN`

   .. py:method:: forward(*args, **kwargs)


   .. py:method:: predict_step(batch, batch_idx, dataloader_idx = 0)



