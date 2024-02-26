:py:mod:`chemprop.chemprop.featurizers.molgraph.molgraph`
=========================================================

.. py:module:: chemprop.chemprop.featurizers.molgraph.molgraph


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molgraph.molgraph.MolGraph




.. py:class:: MolGraph


   Bases: :py:obj:`NamedTuple`

   A :class:`MolGraph` represents the graph featurization of a molecule.

   .. py:attribute:: V
      :type: numpy.ndarray

      an array of shape ``V x d_v`` containing the atom features of the molecule

   .. py:attribute:: E
      :type: numpy.ndarray

      an array of shape ``E x d_e`` containing the bond features of the molecule

   .. py:attribute:: edge_index
      :type: numpy.ndarray

      an array of shape ``2 x E`` containing the edges of the graph in COO format

   .. py:attribute:: rev_edge_index
      :type: numpy.ndarray

      A array of shape ``E`` that maps from an edge index to the index of the source of the reverse edge in :attr:`edge_index` attribute.


