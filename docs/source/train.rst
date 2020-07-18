.. _train_predict:

Training and Predicting
=======================

`chemprop.train <https://github.com/chemprop/chemprop/tree/master/chemprop/train>`_ contains functions to train and make predictions with message passing neural networks.

Train
-----

`chemprop.train.train.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/train.py>`_ trains a model for a single epoch.

.. automodule:: chemprop.train.train
   :members:


Run Training
------------

`chemprop.train.run_training.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/run_training.py>`_ loads data, initializes the model, and runs training, validation, and testing of the model.

.. automodule:: chemprop.train.run_training
   :members:

Cross-Validation
----------------

`chemprop.train.cross_validate.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/cross_validate.py>`_ provides an outer loop around `chemprop.train.run_training.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/run_training.py>`_ that runs training and evaluating for each of several splits of the data.

.. automodule:: chemprop.train.cross_validate
   :members:

Predict
-------

`chemprop.train.predict.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/predict.py>`_ uses a trained model to make predicts on data.

.. automodule:: chemprop.train.predict
   :members:

Make Predictions
----------------

`chemprop.train.make_predictions.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/make_predicts.py>`_ is a wrapper aoround `chemprop.train.predict.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/predict.py>`_ which loads data, loads a trained model, makes predictions, and saves those predictions.

.. automodule:: chemprop.train.make_predictions
   :members:

Evaluate
--------

`chemprop.train.evaluate.py <https://github.com/chemprop/chemprop/tree/master/chemprop/train/evaluate.py>`_ contains functions for evaluating the quality of predictions by comparing them to the true values.

.. automodule:: chemprop.train.evaluate
   :members:
