:py:mod:`models.modules.message_passing.base`
=============================================

.. py:module:: models.modules.message_passing.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.message_passing.base.MessagePassingProto
   models.modules.message_passing.base.MessagePassingBlock




.. py:class:: MessagePassingProto


   Bases: :py:obj:`Protocol`

   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing), for example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto(Protocol[T]):
           def meth(self) -> T:
               ...

   .. py:attribute:: input_dim
      :type: int

      

   .. py:attribute:: output_dim
      :type: int

      

   .. py:method:: forward(bmg, V_d = None)

      Encode a batch of molecular graphs.

      :param bmg: the batch of :class:`~chemprop.v2.featurizers.molgraph.MolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`,
                  so if provided, this tensor must be 0-padded in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape `V x d_h` or `V x (d_h + d_vd)` containing the hidden representation
                of each vertex in the batch of graphs. The feature dimension depends on whether
                additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: MessagePassingBlock(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`MessagePassingProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   A :class:`MessagePassingBlock` is encodes a batch of molecular graphs using message passing
   to learn vertex-level hidden representations.


