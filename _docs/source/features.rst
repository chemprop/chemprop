.. _features:

Features
========

`chemprop.features <https://github.com/chemprop/chemprop/tree/master/chemprop/features>`_ contains functions for featurizing molecules. This includes both atom/bond features used in message passing and additional molecule-level features appended after message passing.

Featurization
-------------

Classes and functions from `chemprop.features.featurization.py <https://github.com/chemprop/chemprop/tree/master/chemprop/features/featurization.py>`_. Featurization specifically includes computation of the atom and bond features used in message passing.

.. automodule:: chemprop.features.featurization
   :members:

Features Generators
-------------------

Classes and functions from `chemprop.features.features_generators.py <https://github.com/chemprop/chemprop/tree/master/chemprop/features/features_generators.py>`_. Features generators are used for computing additional molecule-level features that are appended after message passing.

.. automodule:: chemprop.features.features_generators
   :members:

Utils
-----

Classes and functions from `chemprop.features.utils.py <https://github.com/chemprop/chemprop/tree/master/chemprop/features/utils.py>`_.

.. automodule:: chemprop.features.utils
   :members:
