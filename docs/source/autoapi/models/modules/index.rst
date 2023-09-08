:py:mod:`models.modules`
========================

.. py:module:: models.modules


Subpackages
-----------
.. toctree::
   :titlesonly:
   :maxdepth: 3

   message_passing/index.rst


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   agg/index.rst
   ffn/index.rst
   readout/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.Aggregation
   models.modules.MeanAggregation
   models.modules.SumAggregation
   models.modules.NormAggregation
   models.modules.MessagePassingBlock
   models.modules.MessagePassingBlockBase
   models.modules.AtomMessageBlock
   models.modules.BondMessageBlock
   models.modules.MulticomponentMessagePassing
   models.modules.Readout
   models.modules.ReadoutFFNBase
   models.modules.RegressionFFN
   models.modules.MveFFN
   models.modules.EvidentialFFN
   models.modules.BinaryClassificationFFNBase
   models.modules.BinaryClassificationFFN
   models.modules.BinaryDirichletFFN
   models.modules.MulticlassClassificationFFN
   models.modules.MulticlassDirichletFFN
   models.modules.SpectralFFN




Attributes
~~~~~~~~~~

.. autoapisummary::

   models.modules.AggregationRegistry
   models.modules.ReadoutRegistry


.. py:data:: AggregationRegistry

   

.. py:class:: Aggregation(dim = 0)


   Bases: :py:obj:`abc.ABC`, :py:obj:`torch.nn.Module`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   An :class:`Aggregation` aggregates the node-level representations of a batch of graphs into
   a batch of graph-level representations

   **NOTE**: this class is abstract and cannot be instantiated. Instead, you must use one of the
   concrete subclasses.

   .. seealso:: :class:`chemprop.v2.models.modules.agg.MeanAggregation`, :class:`chemprop.v2.models.modules.agg.SumAggregation`, :class:`chemprop.v2.models.modules.agg.NormAggregation`

   .. py:method:: forward(H, sizes)

      Aggregate the graph-level representations of a batch of graphs into their respective
      global representations

      NOTE: it is possible for a graph to have 0 nodes. In this case, the representation will be
      a zero vector of length `d` in the final output.

      :param H: A tensor of shape ``sum(sizes) x d`` containing the stacked node-level representations
                of ``len(sizes)`` graphs
      :type H: Tensor
      :param sizes: an list containing the number of nodes in each graph, respectively.
      :type sizes: Sequence[int]

      :returns: a tensor of shape ``len(sizes) x d`` containing the graph-level representation of each
                graph
      :rtype: Tensor

      :raises ValueError: if ``sum(sizes)`` is not equal to ``len(H_v)``

      .. rubric:: Examples

      **NOTE**: the following examples are for illustrative purposes only. In practice, you must
      use one of the concrete subclasses.

      1. A typical use-case:

      >>> H = torch.rand(10, 4)
      >>> sizes = [3, 4, 3]
      >>> agg = Aggregation()
      >>> agg(H, sizes).shape
      torch.Size([3, 4])

      2. A batch containing a graph with 0 nodes:

      >>> H = torch.rand(10, 4)
      >>> sizes = [3, 4, 0, 3]
      >>> agg = Aggregation()
      >>> agg(H, sizes).shape
      torch.Size([4, 4])


   .. py:method:: agg(H)
      :abstractmethod:

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: MeanAggregation(dim = 0)


   Bases: :py:obj:`Aggregation`

   Average the graph-level representation

   .. math::
       \mathbf h = \frac{1}{|V|} \sum_{v \in V} \mathbf h_v

   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: SumAggregation(dim = 0)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation

   .. math::
       \mathbf h = \sum_{v \in V} \mathbf h_v


   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: NormAggregation(*args, norm = 100, **kwargs)


   Bases: :py:obj:`Aggregation`

   Sum the graph-level representation and divide by a normalization constant

   .. math::
       \mathbf h = \frac{1}{c} \sum_{v \in V} \mathbf h_v

   .. py:method:: agg(H)

      Aggregate the graph-level of a single graph into a vector

      :param H: A tensor of shape ``V x d`` containing the node-level representation of a graph with
                ``V`` nodes and node feature dimension ``d``
      :type H: Tensor

      :returns: a tensor of shape ``d`` containing the global representation of the input graph
      :rtype: Tensor



.. py:class:: MessagePassingBlock(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`MessagePassingProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   A :class:`MessagePassingBlock` is encodes a batch of molecular graphs using message passing
   to learn vertex-level hidden representations.


.. py:class:: MessagePassingBlockBase(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`chemprop.v2.models.modules.message_passing.base.MessagePassingBlock`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   The base message-passing block for atom- and bond-based message-passing schemes

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


   .. py:method:: finalize(M_v, V, V_d)

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


   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)
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


   .. py:method:: forward(bmg, V_d = None)
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



.. py:class:: AtomMessageBlock(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   A :class:`AtomMessageBlock` encodes a batch of molecular graphs by passing messages along
   atoms.

   It implements the following operation:

   .. math::

       h_v^{(0)} &= \tau \left( \mathbf{W}_i(x_v) \right) \\
       m_v^{(t)} &= \sum_{u \in \mathcal{N}(v)} h_u^{(t-1)} \mathbin\Vert e_{uv} \\
       h_v^{(t)} &= \tau\left(h_v^{(0)} + \mathbf{W}_h m_v^{(t-1)}\right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf{W}_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right)  \right),

   where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h`, and
   :math:`\mathbf{W}_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_v^{(t)}` is the hidden representation of atom :math:`v` at iteration :math:`t`;
   :math:`m_v^{(t)}` is the message received by atom :math:`v` at iteration :math:`t`; and
   :math:`t \in \{1, \dots, T\}` is the number of message passing iterations.

   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

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


   .. py:method:: forward(bmg, V_d = None)

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



.. py:class:: BondMessageBlock(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`MessagePassingBlockBase`

   A :class:`BondMessageBlock` encodes a batch of molecular graphs by passing messages along
   directed bonds.

   It implements the following operation:

   .. math::

       h_{vw}^{(0)} &= \tau \left( \mathbf{W}_i(e_{vw}) \right) \\
       m_{vw}^{(t)} &= \sum_{u \in \mathcal{N}(v)\setminus w} h_{uv}^{(t-1)} \\
       h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf{W}_h m_{vw}^{(t-1)} \right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal{N}(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf{W}_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

   where :math:`\tau` is the activation function; :math:`\mathbf{W}_i`, :math:`\mathbf{W}_h`, and
   :math:`\mathbf{W}_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at
   iteration :math:`t`; :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v
   \rightarrow w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of
   message passing iterations.

   .. py:method:: build(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

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


   .. py:method:: forward(bmg, V_d = None)

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



.. py:data:: ReadoutRegistry

   

.. py:class:: Readout(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`_ReadoutProto`, :py:obj:`chemprop.v2.models.hparams.HasHParams`

   A :class:`Readout` is a protocol that defines a fully differentiable function which maps a tensor of shape `N x d_i` to a tensor of shape `N x d_o`


.. py:class:: ReadoutFFNBase(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`Readout`, :py:obj:`lightning.pytorch.core.mixins.HyperparametersMixin`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:property:: input_dim
      :type: int


   .. py:property:: output_dim
      :type: int


   .. py:property:: n_tasks
      :type: int


   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: RegressionFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MveFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: EvidentialFFN(*args, loc = 0, scale = 1, **kwargs)


   Bases: :py:obj:`RegressionFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 4

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryClassificationFFNBase(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.


.. py:class:: BinaryClassificationFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: BinaryDirichletFFN(n_tasks = 1, input_dim = DEFAULT_HIDDEN_DIM, hidden_dim = 300, n_layers = 1, dropout = 0, activation = 'relu', criterion = None)


   Bases: :py:obj:`BinaryClassificationFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 2

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassClassificationFFN(n_classes, n_tasks = 1, *args, **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: MulticlassDirichletFFN(n_classes, n_tasks = 1, *args, **kwargs)


   Bases: :py:obj:`MulticlassClassificationFFN`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:method:: forward(Z)


   .. py:method:: train_step(Z)



.. py:class:: SpectralFFN(*args, spectral_activation = 'softplus', **kwargs)


   Bases: :py:obj:`ReadoutFFNBase`

   A :class:`ReadoutFFNBase` is the base class for all readout functions that use a
   :class:`SimpleFFN` to map the learned fingerprint to the desired output.

   .. py:attribute:: n_targets
      :value: 1

      


