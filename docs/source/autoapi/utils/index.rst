:py:mod:`utils`
===============

.. py:module:: utils


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   mixins/index.rst
   registry/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   utils.ReprMixin
   utils.ClassRegistry
   utils.AutoName



Functions
~~~~~~~~~

.. autoapisummary::

   utils.make_mol
   utils.pretty_shape



.. py:class:: ReprMixin


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: get_params()



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



.. py:class:: AutoName(*args, **kwds)


   Bases: :py:obj:`enum.Enum`

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

   .. py:method:: __str__()

      Return str(self).


   .. py:method:: get(name)
      :classmethod:


   .. py:method:: keys()
      :classmethod:



.. py:function:: make_mol(smi, keep_h, add_h)

   build an RDKit molecule from a SMILES string.

   :param smi: a SMILES string.
   :type smi: str
   :param keep_h: whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified
   :type keep_h: bool
   :param add_h: whether to add hydrogens to the molecule
   :type add_h: bool

   :returns: the RDKit molecule.
   :rtype: Chem.Mol


.. py:function:: pretty_shape(shape)

   Make a pretty string from an input shape

   .. rubric:: Example

   >>> X = np.random.rand(10, 4)
   >>> X.shape
   (10, 4)
   >>> pretty_shape(X.shape)
   '10 x 4'


