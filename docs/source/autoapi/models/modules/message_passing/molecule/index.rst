:py:mod:`models.modules.message_passing.molecule`
=================================================

.. py:module:: models.modules.message_passing.molecule


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.message_passing.molecule.MessagePassingBlockBase
   models.modules.message_passing.molecule.BondMessageBlock
   models.modules.message_passing.molecule.AtomMessageBlock




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



