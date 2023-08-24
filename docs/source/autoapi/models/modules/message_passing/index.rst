:py:mod:`models.modules.message_passing`
========================================

.. py:module:: models.modules.message_passing


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   base/index.rst
   molecule/index.rst
   multi/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.message_passing.MessagePassingBlock
   models.modules.message_passing.MessagePassingBlockBase
   models.modules.message_passing.AtomMessageBlock
   models.modules.message_passing.BondMessageBlock
   models.modules.message_passing.MulticomponentMessagePassing




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


.. py:class:: MessagePassingBlockBase(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, bias: bool = False, depth: int = 3, dropout: float = 0, activation: str = 'relu', undirected: bool = False, d_vd: int | None = None)


   Bases: :py:obj:`chemprop.v2.models.modules.message_passing.base.MessagePassingBlock`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   The base message-passing block for atom- and bond-based MPNNs

   NOTE: this class is an abstract base class and cannot be instantiated

   :param d_v: the feature dimension of the vertices
   :type d_v: int, default=DEFAULT_ATOM_FDIM
   :param d_e: the feature dimension of the edges
   :type d_e: int, default=DEFAULT_BOND_FDIM
   :param d_h: the hidden dimension during message passing
   :type d_h: int, default=DEFAULT_HIDDEN_DIM
   :param bias: if `True`, add a bias term to the learned weight matrices
   :type bias: bool, defuault=False
   :param depth: the number of message passing iterations
   :type depth: int, default=3
   :param undirected: if `True`, pass messages on undirected edges
   :type undirected: bool, default=False
   :param dropout: the dropout probability
   :type dropout: float, default=0
   :param activation: the activation function to use
   :type activation: str, default="relu"
   :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout
   :type d_vd: int | None, default=None

   .. seealso::

      * :class:`AtomMessageBlock`

      * :class:`BondMessageBlock`

   .. py:property:: output_dim
      :type: int


   .. py:method:: finalize(M_v: torch.Tensor, V: torch.Tensor, V_d: torch.Tensor | None) -> torch.Tensor

      Finalize message passing by (1) concatenating the final hidden representations `H_v`
      and the original vertex ``V`` and (2) further concatenating additional vertex descriptors
      ``V_d``, if provided.

      This function implements the following operation:

      .. math::
          H_v &= \mathtt{dropout} \left( \tau(\mathbf{W}_o(V \mathbin\Vert M_v)) \right) \\
          H_v &= \mathtt{dropout} \left( \tau(\mathbf{W}_d(H_v \mathbin\Vert V_d)) \right),

      where :math:`\tau` is the activation function, :math:`\Vert` is the concatenation operator,
      :math:`\mathbf{W}_o` and :math:`\mathbf{W}_d` are learned weight matrices, :math:`M_v` is
      the message matrix, :math:`V` is the original vertex feature matrix, and :math:`V_d` is an
      optional vertex descriptor matrix.

      :param M_v: a tensor of shape ``V x d_h`` containing the messages sent from each atom
      :type M_v: Tensor
      :param V: a tensor of shape ``V x d_v`` containing the original vertex features
      :type V: Tensor
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional vertex descriptors
      :type V_d: Tensor | None

      :returns: a tensor of shape ``V x (d_h + d_v [+ d_vd])`` containing the final hidden
                representations
      :rtype: Tensor

      :raises InvalidShapeError: if ``V_d`` is not of shape ``b x d_vd``, where ``b`` is the batch size and ``d_vd`` is
          the vertex descriptor dimension


   .. py:method:: build(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, d_vd: int | None = None, bias: bool = False) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module | None]
      :abstractmethod:

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg: chemprop.v2.featurizers.BatchMolGraph, V_d: torch.Tensor | None = None) -> torch.Tensor
      :abstractmethod:

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: AtomMessageBlock(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, bias: bool = False, depth: int = 3, dropout: float = 0, activation: str = 'relu', undirected: bool = False, d_vd: int | None = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   The base message-passing block for atom- and bond-based MPNNs

   NOTE: this class is an abstract base class and cannot be instantiated

   :param d_v: the feature dimension of the vertices
   :type d_v: int, default=DEFAULT_ATOM_FDIM
   :param d_e: the feature dimension of the edges
   :type d_e: int, default=DEFAULT_BOND_FDIM
   :param d_h: the hidden dimension during message passing
   :type d_h: int, default=DEFAULT_HIDDEN_DIM
   :param bias: if `True`, add a bias term to the learned weight matrices
   :type bias: bool, defuault=False
   :param depth: the number of message passing iterations
   :type depth: int, default=3
   :param undirected: if `True`, pass messages on undirected edges
   :type undirected: bool, default=False
   :param dropout: the dropout probability
   :type dropout: float, default=0
   :param activation: the activation function to use
   :type activation: str, default="relu"
   :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout
   :type d_vd: int | None, default=None

   .. seealso::

      * :class:`AtomMessageBlock`

      * :class:`BondMessageBlock`

   .. py:method:: build(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, d_vd: int | None = None, bias: bool = False)

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg: chemprop.v2.featurizers.BatchMolGraph, V_d: torch.Tensor | None = None) -> torch.Tensor

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



.. py:class:: BondMessageBlock(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, bias: bool = False, depth: int = 3, dropout: float = 0, activation: str = 'relu', undirected: bool = False, d_vd: int | None = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   The base message-passing block for atom- and bond-based MPNNs

   NOTE: this class is an abstract base class and cannot be instantiated

   :param d_v: the feature dimension of the vertices
   :type d_v: int, default=DEFAULT_ATOM_FDIM
   :param d_e: the feature dimension of the edges
   :type d_e: int, default=DEFAULT_BOND_FDIM
   :param d_h: the hidden dimension during message passing
   :type d_h: int, default=DEFAULT_HIDDEN_DIM
   :param bias: if `True`, add a bias term to the learned weight matrices
   :type bias: bool, defuault=False
   :param depth: the number of message passing iterations
   :type depth: int, default=3
   :param undirected: if `True`, pass messages on undirected edges
   :type undirected: bool, default=False
   :param dropout: the dropout probability
   :type dropout: float, default=0
   :param activation: the activation function to use
   :type activation: str, default="relu"
   :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden features before readout
   :type d_vd: int | None, default=None

   .. seealso::

      * :class:`AtomMessageBlock`

      * :class:`BondMessageBlock`

   .. py:method:: build(d_v: int = DEFAULT_ATOM_FDIM, d_e: int = DEFAULT_BOND_FDIM, d_h: int = DEFAULT_HIDDEN_DIM, d_vd: int | None = None, bias: bool = False)

      construct the weight matrices used in the message passing update functions

      :param d_v: the vertex feature dimension
      :type d_v: int
      :param d_e: the edge feature dimension
      :type d_e: int
      :param d_h: the hidden dimension during message passing
      :type d_h: int, default=300
      :param d_vd: the dimension of additional vertex descriptors that will be concatenated to the hidden
                   features before readout, if any
      :type d_vd: int | None, default=None
      :param bias: whether to add a learned bias to the matrices
      :type bias: bool, default=False

      :returns: **W_i, W_h, W_o, W_d** -- the input, hidden, output, and descriptor weight matrices, respectively, used in the
                message passing update functions. The descriptor weight matrix is `None` if no vertex
                dimension is supplied
      :rtype: tuple[nn.Module, nn.Module, nn.Module, nn.Module | None]


   .. py:method:: forward(bmg: chemprop.v2.featurizers.BatchMolGraph, V_d: torch.Tensor | None = None) -> torch.Tensor

      Encode a batch of molecular graphs.

      :param bmg: a batch of :class:`BatchMolGraph`s to encode
      :type bmg: BatchMolGraph
      :param V_d: an optional tensor of shape ``V x d_vd`` containing additional descriptors for each atom
                  in the batch. These will be concatenated to the learned atomic descriptors and
                  transformed before the readout phase.
                  **NOTE**: recall that ``V`` is equal to ``num_atoms + 1``, so ``V_d`` must be 0-padded
                  in the 0th row.
      :type V_d: Tensor | None, default=None

      :returns: a tensor of shape ``b x d_h`` or ``b x (d_h + d_vd)`` containing the encoding of each
                molecule in the batch, depending on whether additional atom descriptors were provided
      :rtype: Tensor



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



