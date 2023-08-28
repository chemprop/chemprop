:py:mod:`utils.registry`
========================

.. py:module:: utils.registry


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   utils.registry.ClassRegistry
   utils.registry.Factory




Attributes
~~~~~~~~~~

.. autoapisummary::

   utils.registry.T


.. py:data:: T

   

.. py:class:: ClassRegistry


   Bases: :py:obj:`dict`\ [\ :py:obj:`Any`\ , :py:obj:`Type`\ [\ :py:obj:`T`\ ]\ ]

   dict() -> new empty dictionary
   dict(mapping) -> new dictionary initialized from a mapping object's
       (key, value) pairs
   dict(iterable) -> new dictionary initialized as if via:
       d = {}
       for k, v in iterable:
           d[k] = v
   dict(**kwargs) -> new dictionary initialized with the name=value pairs
       in the keyword argument list.  For example:  dict(one=1, two=2)

   .. py:attribute:: __call__

      

   .. py:method:: register(alias = None)


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: __str__()

      Return str(self).



.. py:class:: Factory


   .. py:method:: build(clz_T, *args, **kwargs)
      :classmethod:



