:py:mod:`featurizers.utils`
===========================

.. py:module:: featurizers.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   featurizers.utils.MultiHotFeaturizerMixin
   featurizers.utils.ReactionMode




.. py:class:: MultiHotFeaturizerMixin


   A `MultiHotFeaturizer` calculates feature vectors of arbitrary objects by concatenating
   multiple one-hot feature vectors

   .. py:property:: num_subfeatures
      :type: int


   .. py:method:: __call__(x) -> numpy.ndarray


   .. py:method:: one_hot_index(x, xs: Sequence) -> tuple[int, int]
      :staticmethod:

      the index of `x` in `xs`, if it exists. Otherwise, return `len(xs) + 1`.



.. py:class:: ReactionMode(*args, **kwds)


   Bases: :py:obj:`chemprop.v2.utils.AutoName`

   The manner in which a reaction should be featurized into a `MolGraph`

   .. py:attribute:: REAC_PROD

      concatenate the reactant features with the product features.

   .. py:attribute:: REAC_PROD_BALANCE

      concatenate the reactant features with the products feature and balances imbalanced
      reactions

   .. py:attribute:: REAC_DIFF

      concatenates the reactant features with the difference in features between reactants and
      products

   .. py:attribute:: REAC_DIFF_BALANCE

      concatenates the reactant features with the difference in features between reactants and
      product and balances imbalanced reactions

   .. py:attribute:: PROD_DIFF

      concatenates the product features with the difference in features between reactants and
      products

   .. py:attribute:: PROD_DIFF_BALANCE

      concatenates the product features with the difference in features between reactants and
      products and balances imbalanced reactions


