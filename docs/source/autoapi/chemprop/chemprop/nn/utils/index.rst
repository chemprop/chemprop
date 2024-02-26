:py:mod:`chemprop.chemprop.nn.utils`
====================================

.. py:module:: chemprop.chemprop.nn.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.utils.Activation



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.nn.utils.get_activation_function



.. py:class:: Activation


   Bases: :py:obj:`chemprop.utils.utils.EnumMapping`

   Enum where members are also (and must be) strings

   .. py:attribute:: RELU

      

   .. py:attribute:: LEAKYRELU

      

   .. py:attribute:: PRELU

      

   .. py:attribute:: TANH

      

   .. py:attribute:: SELU

      

   .. py:attribute:: ELU

      


.. py:function:: get_activation_function(activation)

   Gets an activation function module given the name of the activation.

   See :class:`~chemprop.v2.models.utils.Activation` for available activations.

   :param activation: The name of the activation function.
   :type activation: str | Activation

   :returns: The activation function module.
   :rtype: nn.Module


