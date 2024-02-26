:py:mod:`chemprop.chemprop.utils.utils`
=======================================

.. py:module:: chemprop.chemprop.utils.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.utils.utils.EnumMapping



Functions
~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.utils.utils.make_mol
   chemprop.chemprop.utils.utils.pretty_shape



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


