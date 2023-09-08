:py:mod:`models.utils`
======================

.. py:module:: models.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   models.utils.Activation



Functions
~~~~~~~~~

.. autoapisummary::

   models.utils.get_activation_function



.. py:class:: Activation(*args, **kwds)


   Bases: :py:obj:`chemprop.v2.utils.utils.AutoName`

   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

   >>> Color.RED
   <Color.RED: 1>

   - value lookup:

   >>> Color(1)
   <Color.RED: 1>

   - name lookup:

   >>> Color['RED']
   <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.

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


