:py:mod:`chemprop.chemprop.nn.message_passing.base`
===================================================

.. py:module:: chemprop.chemprop.nn.message_passing.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.message_passing.base.BondMessagePassing
   chemprop.chemprop.nn.message_passing.base.AtomMessagePassing




.. py:class:: BondMessagePassing(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`_MessagePassingBase`

   A :class:`BondMessagePassing` encodes a batch of molecular graphs by passing messages along
   directed bonds.

   It implements the following operation:

   .. math::

       h_{vw}^{(0)} &= \tau \left( \mathbf W_i(e_{vw}) \right) \\
       m_{vw}^{(t)} &= \sum_{u \in \mathcal N(v)\setminus w} h_{uv}^{(t-1)} \\
       h_{vw}^{(t)} &= \tau \left(h_v^{(0)} + \mathbf W_h m_{vw}^{(t-1)} \right) \\
       m_v^{(T)} &= \sum_{w \in \mathcal N(v)} h_w^{(T-1)} \\
       h_v^{(T)} &= \tau \left (\mathbf W_o \left( x_v \mathbin\Vert m_{v}^{(T)} \right) \right),

   where :math:`\tau` is the activation function; :math:`\mathbf W_i`, :math:`\mathbf W_h`, and
   :math:`\mathbf W_o` are learned weight matrices; :math:`e_{vw}` is the feature vector of the
   bond between atoms :math:`v` and :math:`w`; :math:`x_v` is the feature vector of atom :math:`v`;
   :math:`h_{vw}^{(t)}` is the hidden representation of the bond :math:`v \rightarrow w` at
   iteration :math:`t`; :math:`m_{vw}^{(t)}` is the message received by the bond :math:`v
   \to w` at iteration :math:`t`; and :math:`t \in \{1, \dots, T-1\}` is the number of
   message passing iterations.

   .. py:method:: setup(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      setup the weight matrices used in the message passing update functions

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


   .. py:method:: initialize(bmg)

      initialize the message passing scheme by calculating initial matrix of hidden features


   .. py:method:: message(H, bmg)

      Calculate the message matrix



.. py:class:: AtomMessagePassing(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, bias = False, depth = 3, dropout = 0, activation = Activation.RELU, undirected = False, d_vd = None)


   Bases: :py:obj:`_MessagePassingBase`

   A :class:`AtomMessagePassing` encodes a batch of molecular graphs by passing messages along
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

   .. py:method:: setup(d_v = DEFAULT_ATOM_FDIM, d_e = DEFAULT_BOND_FDIM, d_h = DEFAULT_HIDDEN_DIM, d_vd = None, bias = False)

      setup the weight matrices used in the message passing update functions

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


   .. py:method:: initialize(bmg)

      initialize the message passing scheme by calculating initial matrix of hidden features


   .. py:method:: message(H, bmg)

      Calculate the message matrix



