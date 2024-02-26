:py:mod:`chemprop.chemprop.data.datapoints`
===========================================

.. py:module:: chemprop.chemprop.data.datapoints


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.data.datapoints.MoleculeDatapoint
   chemprop.chemprop.data.datapoints.ReactionDatapoint




.. py:class:: MoleculeDatapoint


   Bases: :py:obj:`_DatapointMixin`, :py:obj:`_MoleculeDatapointMixin`

   A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets.

   .. py:attribute:: V_f
      :type: numpy.ndarray | None

      a numpy array of shape ``V x d_vf``, where ``V`` is the number of atoms in the molecule, and
      ``d_vf`` is the number of additional features that will be concatenated to atom-level features
      *before* message passing

   .. py:attribute:: E_f
      :type: numpy.ndarray | None

      A numpy array of shape ``E x d_ef``, where ``E`` is the number of bonds in the molecule, and
      ``d_ef`` is the number of additional features  containing additional features that will be
      concatenated to bond-level features *before* message passing

   .. py:attribute:: V_d
      :type: numpy.ndarray | None

      A numpy array of shape ``V x d_vd``, where ``V`` is the number of atoms in the molecule, and
      ``d_vd`` is the number of additional features that will be concatenated to atom-level features
      *after* message passing

   .. py:method:: __post_init__(mfs)


   .. py:method:: __len__()


   .. py:method:: calc_features(mfs)



.. py:class:: ReactionDatapoint


   Bases: :py:obj:`_DatapointMixin`, :py:obj:`_ReactionDatapointMixin`

   A :class:`ReactionDatapoint` contains a single reaction and its associated features and targets.

   .. py:method:: __post_init__(mfs)


   .. py:method:: __len__()


   .. py:method:: calc_features(mfs)



