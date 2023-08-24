:py:mod:`models.modules.message_passing.multi`
==============================================

.. py:module:: models.modules.message_passing.multi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.message_passing.multi.MulticomponentMessagePassing




.. py:class:: MulticomponentMessagePassing(blocks: Sequence[chemprop.v2.models.modules.message_passing.molecule.MessagePassingBlockBase], n_components: int, shared: bool = False)


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


   .. py:method:: __len__() -> int


   .. py:method:: forward(bmgs: Iterable[chemprop.v2.featurizers.molgraph.BatchMolGraph], V_ds: Iterable[torch.Tensor | None]) -> torch.Tensor

      Encode the multicomponent inputs

      :param bmgs:
      :type bmgs: Iterable[BatchMolGraph]
      :param V_ds:
      :type V_ds: Iterable[Tensor | None]

      :returns: a list of tensors of shape `b x d_i` containing the respective encodings of the `i`th component, where `b` is the number of components in the batch, and `d_i` is the output dimension of the `i`th encoder
      :rtype: list[Tensor]



