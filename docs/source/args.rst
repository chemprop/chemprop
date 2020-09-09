.. _args:

Command Line Arguments
======================

`chemprop.args.py <https://github.com/chemprop/chemprop/tree/master/chemprop/args.py>`_ contains all command line arguments, which are processed using the `Typed Argument Parser <https://github.com/swansonk14/typed-argument-parser>`_ (:code:`Tap`) package.

Common Arguments
----------------

.. autoclass:: chemprop.args.CommonArgs
   :members:

Train Arguments
---------------

.. autoclass:: chemprop.args.TrainArgs
   :members:

Predict Arguments
-----------------

.. autoclass:: chemprop.args.PredictArgs
   :members:

Interpret Arguments
-------------------

.. autoclass:: chemprop.args.InterpretArgs
   :members:

Hyperparameter Optimization Arguments
-------------------------------------

.. autoclass:: chemprop.args.HyperoptArgs
   :members:

Scikit-Learn Train Arguments
----------------------------

.. autoclass:: chemprop.args.SklearnTrainArgs
   :members:

Scikit-Learn Predict Arguments
------------------------------

.. autoclass:: chemprop.args.SklearnPredictArgs
   :members:

Utility Functions
-----------------

.. automodule:: chemprop.args
   :members: get_checkpoint_paths
