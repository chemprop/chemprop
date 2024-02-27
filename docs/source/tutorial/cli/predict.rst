.. _predict:

Prediction
----------

To load a trained model and make predictions, run ``chemprop predict`` and specify:

* ``--test-path <path>`` Path to the data to predict on.
* A checkpoint by using either:

  #. ``--checkpoint-dir <dir>`` Directory where the model checkpoint(s) are saved (i.e. ``--save_dir`` during training). This will walk the directory, load all ``.pt`` files it finds, and treat the models as an ensemble.
  #. ``--checkpoint-path <path>`` Path to a model checkpoint file (``.pt`` file).

* ``--preds-path`` Path where a CSV file containing the predictions will be saved.

Multiple checkpoints can also be specified using the keyword ``--checkpoint-paths``.

For example:

.. code-block::

    chemprop predict --test-path data/tox21.csv \
        --checkpoint-dir tox21_checkpoints \
        --preds-path tox21_preds.csv

or

.. code-block::

    chemprop predict --test-path data/tox21.csv \
        --checkpoint-path tox21_checkpoints/fold_0/model_0/model.pt \
        --preds-path tox21_preds.csv

.. warning:: 
    This page is under construction.