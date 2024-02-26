:py:mod:`chemprop.chemprop.featurizers.molecule`
================================================

.. py:module:: chemprop.chemprop.featurizers.molecule


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molecule.MoleculeFeaturizer
   chemprop.chemprop.featurizers.molecule.MorganFeaturizerMixin
   chemprop.chemprop.featurizers.molecule.BinaryFeaturizerMixin
   chemprop.chemprop.featurizers.molecule.CountFeaturizerMixin
   chemprop.chemprop.featurizers.molecule.MorganBinaryFeaturzer
   chemprop.chemprop.featurizers.molecule.MorganCountFeaturizer




Attributes
~~~~~~~~~~

.. autoapisummary::

   chemprop.chemprop.featurizers.molecule.MoleculeFeaturizerRegistry


.. py:class:: MoleculeFeaturizer


   Bases: :py:obj:`abc.ABC`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.

   .. py:method:: __len__()
      :abstractmethod:

      the length of the feature vector


   .. py:method:: __call__(mol)
      :abstractmethod:

      Featurize the molecule ``mol``



.. py:data:: MoleculeFeaturizerRegistry

   

.. py:class:: MorganFeaturizerMixin(radius = 2, length = 2048, include_chirality = True)


   .. py:method:: __len__()



.. py:class:: BinaryFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: CountFeaturizerMixin


   .. py:method:: __call__(mol)



.. py:class:: MorganBinaryFeaturzer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`BinaryFeaturizerMixin`, :py:obj:`MoleculeFeaturizer`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.


.. py:class:: MorganCountFeaturizer(radius = 2, length = 2048, include_chirality = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`CountFeaturizerMixin`, :py:obj:`MoleculeFeaturizer`

   A :class:`MoleculeFeaturizer` calculates feature vectors of RDKit molecules.


