:py:mod:`chemprop.chemprop.nn.message_passing.proto`
====================================================

.. py:module:: chemprop.chemprop.nn.message_passing.proto


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.message_passing.proto.MessagePassing




.. py:class:: MessagePassing(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`chemprop.nn.hparams.HasHParams`

   A :class:`MessagePassing` module encodes a batch of molecular graphs
   using message passing to learn vertex-level hidden representations.

   .. py:attribute:: input_dim
      :type: int

      

   .. py:attribute:: output_dim
      :type: int

      

   .. py:method:: forward(bmg, V_d = None)
      :abstractmethod:

      Encode a batch of molecular graphs.

      :param bmg: the batch of :class:`~chemprop.featurizers.molgraph.MolGraph`\s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`\,
                  so if provided, this tensor must be 0-padded in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
                of each vertex in the batch of graphs. The feature dimension depends on whether
                additional atom descriptors were provided
      :rtype: Tensor



