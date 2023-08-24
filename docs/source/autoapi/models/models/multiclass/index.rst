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




.. py:class:: MulticlassMPNN(mp_block: chemprop.v2.models.modules.MessagePassingBlock, n_tasks: int, n_classes: int, *args, **kwargs)


   Bases: :py:obj:`chemprop.v2.models.models.base.MPNN`

   .. py:method:: forward(*args, **kwargs) -> torch.Tensor


   .. py:method:: predict_step(batch: chemprop.v2.data.dataloader.TrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> tuple[torch.Tensor, Ellipsis]



.. py:class:: DirichletMulticlassMPNN(*args, **kwargs)


   Bases: :py:obj:`MulticlassMPNN`

   .. py:method:: forward(*args, **kwargs) -> torch.Tensor


   .. py:method:: predict_step(batch: chemprop.v2.data.dataloader.TrainingBatch, batch_idx: int, dataloader_idx: int = 0) -> tuple[torch.Tensor, Ellipsis]



