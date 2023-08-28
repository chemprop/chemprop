:py:mod:`featurizers.featurizers`
=================================

.. py:module:: featurizers.featurizers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.featurizers.MorganFeaturizerMixin
   featurizers.featurizers.BinaryFeaturizerMixin
   featurizers.featurizers.CountFeaturizerMixin
   featurizers.featurizers.MorganBinaryFeaturzer
   featurizers.featurizers.MorganCountFeaturizer




Attributes
~~~~~~~~~~

.. autoapisummary::

   featurizers.featurizers.MoleculeFeaturizerRegistry


.. py:data:: MoleculeFeaturizerRegistry

   

.. py:class:: MorganFeaturizerMixin(radius = 2, length = 2048, include_chirality = True)


   .. py:method:: __len__()



.. py:class:: BinaryFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: CountFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: MorganBinaryFeaturzer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`BinaryFeaturizerMixin`, :py:obj:`chemprop.v2.featurizers.protos.MoleculeFeaturizerProto`

   A :class:`MoleculeFeaturizerProto` calculates feature vectors of RDKit molecules.


.. py:class:: MorganCountFeaturizer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`CountFeaturizerMixin`, :py:obj:`chemprop.v2.featurizers.protos.MoleculeFeaturizerProto`

   A :class:`MoleculeFeaturizerProto` calculates feature vectors of RDKit molecules.


