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

      

   .. py:method:: forward(bmg: chemprop.v2.featurizers.molgraph.BatchMolGraph, V_d: torch.Tensor | None = None) -> torch.Tensor

      Encode a batch of molecular graphs.

      :param bmg: the batch of `MolGraphs` to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape `V x d_vd` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase. NOTE: recall that `V` is equal to `num_atoms + 1`,
                  so if provided, this tensor must be 0-padded in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape `b x d_h` or `b x (d_h + d_vd)` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: MessagePassingBlock(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`MessagePassingProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


