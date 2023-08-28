:py:mod:`models.modules.message_passing.multi`
==============================================

.. py:module:: models.modules.message_passing.multi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.message_passing.multi.MulticomponentMessagePassing




.. py:class:: MulticomponentMessagePassing(blocks, n_components, shared = False)


   Bases: :py:obj:`torch.nn.Module`

   A `MulticomponentMessagePassing` performs message-passing on each individual input in a
   multicomponent input then concatenates the representation of each input to construct a
   global representation

   :param blocks: the invidual message-passing blocks for each input
   :type blocks: Sequence[MolecularMessagePassingBlock]
   :param n_components: the number of components in each input
   :type n_components: int
   :param shared: whether one block will be shared among all components in an input. If not, a separate
                  block will be learned for each component.
   :type shared: bool, default=False

   .. py:property:: output_dim
      :type: int


   .. py:method:: __len__()


   .. py:method:: forward(bmgs, V_ds)

      Encode the multicomponent inputs

      :param bmgs:
      :type bmgs: Iterable[BatchMolGraph]
      :param V_ds:
      :type V_ds: Iterable[Tensor | None]

      :returns: a list of tensors of shape `b x d_i` containing the respective encodings of the `i`th component, where `b` is the number of components in the batch, and `d_i` is the output dimension of the `i`th encoder
      :rtype: list[Tensor]



