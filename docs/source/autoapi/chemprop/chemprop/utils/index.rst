:py:mod:`chemprop.chemprop.utils`
=================================

.. py:module:: chemprop.chemprop.utils


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   mixins/index.rst
   registry/index.rst
   utils/index.rst
   v1_to_v2/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.utils.ReprMixin
   chemprop.chemprop.utils.ClassRegistry
   chemprop.chemprop.utils.Factory
   chemprop.chemprop.utils.EnumMapping



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.utils.make_mol
   chemprop.chemprop.utils.pretty_shape



.. py:class:: ReprMixin


   .. py:method:: __repr__()

      Return repr(self).


   .. py:method:: get_params()



.. py:class:: ClassRegistry


   Bases: :py:obj:`dict`\ [\ :py:obj:`str`\ , :py:obj:`Type`\ [\ :py:obj:`T`\ ]\ ]

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



.. py:class:: EnumMapping


   Bases: :py:obj:`enum.StrEnum`

   Enum where members are also (and must be) strings

   .. py:method:: get(name)
      :classmethod:


   .. py:method:: keys()
      :classmethod:


   .. py:method:: values()
      :classmethod:


   .. py:method:: items()
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


