:py:mod:`chemprop.chemprop.data.collate`
========================================

.. py:module:: chemprop.chemprop.data.collate


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.collate.BatchMolGraph
   chemprop.chemprop.data.collate.TrainingBatch
   chemprop.chemprop.data.collate.MulticomponentTrainingBatch



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.collate.collate_batch
   chemprop.chemprop.data.collate.collate_multicomponent



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


.. py:class:: MulticomponentTrainingBatch


   Bases: :py:obj:`NamedTuple`

   .. py:attribute:: bmgs
      :type: list[BatchMolGraph]

      

   .. py:attribute:: V_ds
      :type: list[torch.Tensor]

      

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

      


.. py:function:: collate_multicomponent(batches)


