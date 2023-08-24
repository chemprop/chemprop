:py:mod:`featurizers.featurizers`
=================================

.. py:module:: featurizers.featurizers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.featurizers.MoleculeFeaturizerProto
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

   

.. py:class:: MoleculeFeaturizerProto


   Bases: :py:obj:`Protocol`

   A :class:`MoleculeFeaturizerProto` defines the protocol for molecule-level featurization

   .. py:method:: __len__() -> int

      the length of the feature vector


   .. py:method:: __call__(mol: rdkit.Chem.Mol) -> numpy.ndarray

      Featurize the molecule into a vector



.. py:class:: MorganFeaturizerMixin(radius: int = 2, length: int = 2048, include_chirality: bool = True)


   .. py:method:: __len__() -> int



.. py:class:: BinaryFeaturizerMixin


   .. py:method:: __call__(mol: rdkit.Chem.Mol) -> numpy.ndarray



.. py:class:: CountFeaturizerMixin


   .. py:method:: __call__(mol: rdkit.Chem.Mol) -> numpy.ndarray



.. py:class:: MorganBinaryFeaturzer(radius: int = 2, length: int = 2048, include_chirality: bool = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`BinaryFeaturizerMixin`, :py:obj:`MoleculeFeaturizerProto`

   A :class:`MoleculeFeaturizerProto` defines the protocol for molecule-level featurization


.. py:class:: MorganCountFeaturizer(radius: int = 2, length: int = 2048, include_chirality: bool = True)


   Bases: :py:obj:`MorganFeaturizerMixin`, :py:obj:`CountFeaturizerMixin`, :py:obj:`MoleculeFeaturizerProto`

   A :class:`MoleculeFeaturizerProto` defines the protocol for molecule-level featurization


