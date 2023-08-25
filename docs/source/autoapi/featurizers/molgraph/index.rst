:py:mod:`featurizers.molgraph`
==============================

.. py:module:: featurizers.molgraph


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.molgraph.MolGraph
   featurizers.molgraph.BatchMolGraph




.. py:class:: MolGraph


   Bases: :py:obj:`NamedTuple`

   A :class:`MolGraph` represents the graph featurization of a molecule.

   .. py:attribute:: n_atoms
      :type: int

      the number of atoms in the molecule

   .. py:attribute:: n_bonds
      :type: int

      the number of bonds in the molecule

   .. py:attribute:: V
      :type: numpy.ndarray

      an array of shape `V x d_v` containing the atom features of the molecule

   .. py:attribute:: E
      :type: numpy.ndarray

      an array of shape `E x d_e` containing the bond features of the molecule

   .. py:attribute:: a2b
      :type: list[tuple[int]]

      A list of length `V` that maps from an atom index to a list of incoming bond indices.

   .. py:attribute:: b2a
      :type: list[int]

      A list of length `E` that maps from a bond index to the index of the atom the bond
      originates from.

   .. py:attribute:: b2revb
      :type: numpy.ndarray

      A list of length `E` that maps from a bond index to the index of the reverse bond.

   .. py:attribute:: a2a
      :type: list[int] | None

      a mapping from atom index to the indices of connected atoms

   .. py:attribute:: b2b
      :type: numpy.ndarray | None

      a mapping from bond index to the indices of connected bonds


.. py:class:: BatchMolGraph


   A :class:`BatchMolGraph` represents a batch of individual :class:`MolGraph`s.

   It has all the attributes of a `MolGraph` with the addition of `a_scope` and `b_scope`. These
   define the respective atom- and bond-scope of each individual `MolGraph` within the
   `BatchMolGraph`. This class is intended for use with data loading, so it uses
   :obj:`~torch.Tensor`s to store data

   .. py:attribute:: mgs
      :type: dataclasses.InitVar[Sequence[MolGraph]]

      A list of individual :class:`MolGraph`s to be batched together

   .. py:attribute:: n_atoms
      :type: int

      the number of atoms in the batched graph

   .. py:attribute:: n_bonds
      :type: int

      the number of bonds in the batched graph

   .. py:attribute:: V
      :type: torch.Tensor

      the atom feature matrix

   .. py:attribute:: E
      :type: torch.Tensor

      the bond feature matrix

   .. py:attribute:: a2b
      :type: torch.Tensor

      a mapping from atom index to indices of incoming bonds

   .. py:attribute:: b2a
      :type: torch.Tensor

      a mapping from bond index to index of the originating atom

   .. py:attribute:: b2revb
      :type: torch.Tensor

      A mapping from bond index to the index of the reverse bond.

   .. py:attribute:: a2a
      :type: torch.Tensor | None

      a mapping from atom index to the indices of connected atoms

   .. py:attribute:: b2b
      :type: torch.Tensor | None

      a mapping from bond index to the indices of connected bonds

   .. py:attribute:: a_scope
      :type: list[int]

      the number of atoms for each molecule in the batch

   .. py:attribute:: b_scope
      :type: list[int]

      the number of bonds for each molecule in the batch

   .. py:method:: __post_init__(mgs)


   .. py:method:: __len__()

      the number of individual :class:`MolGraph`s in this batch


   .. py:method:: to(device)



