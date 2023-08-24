.. _models:

Models
======

`chemprop.models.py <https://github.com/chemprop/chemprop/tree/master/chemprop/models>`_ contains the core Chemprop message passing neural network.

Model
-----

`chemprop.models.model.py <https://github.com/chemprop/chemprop/tree/master/chemprop/models/model.py>`_ contains the :class:`~chemprop.models.model.MoleculeModel` class, which contains the full Chemprop model. It consists of an :class:`~chemprop.models.mpn.MPN`, which performs message passing, along with a feed-forward neural network which combines the output of the message passing network along with any additional molecule-level features and makes the final property predictions.

.. automodule:: chemprop.models.model
   :members:

MPN
---

`chemprop.models.model.py <https://github.com/chemprop/chemprop/tree/master/chemprop/models/model.py>`_ contains the :class:`~chemprop.models.mpn.MPNEncoder` class, which is the core message passing network, along with a wrapper :class:`~chemprop.models.mpn.MPN` which is used within a :class:`~chemprop.models.model.MoleculeModel`.

.. automodule:: chemprop.models.mpn
   :members:
