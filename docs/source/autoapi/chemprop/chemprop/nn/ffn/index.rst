:py:mod:`chemprop.chemprop.nn.ffn`
==================================

.. py:module:: chemprop.chemprop.nn.ffn


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.ffn.FFN
   chemprop.chemprop.nn.ffn.MLP




.. py:class:: FFN(*args, **kwargs)


   Bases: :py:obj:`torch.nn.Module`

   A :class:`FFN` is a differentiable function
   :math:`f_\theta : \mathbb R^i \mapsto \mathbb R^o`

   .. py:attribute:: input_dim
      :type: int

      

   .. py:attribute:: output_dim
      :type: int

      

   .. py:method:: forward(X)
      :abstractmethod:



.. py:class:: MLP(input_dim, output_dim, hidden_dim = 300, n_layers = 1, dropout = 0.0, activation = 'relu')


   Bases: :py:obj:`torch.nn.Sequential`, :py:obj:`FFN`

   An :class:`MLP` is an FFN that implements the following function:

   .. math::
       \mathbf h_0 &= \mathbf x\,\mathbf W^{(0)} + \mathbf b^{(0)} \\
       \mathbf h_l &= \mathtt{dropout} \left(
           \sigma \left(\,\mathbf h_{l-1}\,\mathbf W^{{l)} \right)
       \right) \\
       \mathbf h_L &= \mathbf h_{L-1} \mathbf W^{{l)} + \mathbf b^{{l)},

   where :math:`\mathbf x` is the input tensor, :math:`\mathbf W^{{l)}` is the learned weight matrix
   for the :math:`l`-th layer, :math:`\mathbf b^{{l)}` is the bias vector for the :math:`l`-th layer,
   :math:`\mathbf h^{{l)}` is the hidden representation at layer :math:`l`, :math:`\sigma` is the
   activation function, and :math:`L` is the number of layers.


