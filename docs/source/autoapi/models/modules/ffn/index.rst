:py:mod:`models.modules.ffn`
============================

.. py:module:: models.modules.ffn


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.modules.ffn.FFN
   models.modules.ffn.SimpleFFN




.. py:class:: FFN(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`, :py:obj:`_FFNProto`

   A :class:`FFN` is a fully differentiable function that maps a tensor of shape ``N x d_i`` to a tensor of shape ``N x d_o``

   :inherited-members:


.. py:class:: SimpleFFN(input_dim, output_dim, hidden_dim = 300, n_layers = 1, dropout = 0.0, activation = 'relu')


   Bases: :py:obj:`FFN`

   A :class:`SimpleFFN` is a simple FFN that implements the following function:

   .. math::
       \mathbf H_0 &= \mathbf X\,\mathbf W_0 + \mathbf b_0 \\
       \mathbf H_l &= \mathtt{dropout} \left(
           \sigma \left(\,\mathbf H_{l-1}\,\mathbf W_l \right)
       \right) \\
       \mathbf H_L &= \mathbf H_{L-1} \mathbf W_L + \mathbf b_L,

   where :math:`\mathbf X` is the input tensor, :math:`\mathbf W_l` is the learned weight matrix
   for the :math:`l`-th layer, :math:`\mathbf b_l` is the bias vector for the :math:`l`-th layer,
   :math:`\mathbf H_l` is the hidden representation at layer :math:`l`, :math:`\sigma` is the
   activation function, and :math:`L` is the number of layers.

   :inherited-members:

   .. py:method:: forward(X)



