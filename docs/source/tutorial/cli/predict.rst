.. _predict:

Prediction
----------

To load a trained model and make predictions, run ``chemprop predict`` and specify:

* ``--test_path <path>`` Path to the data to predict on.
* A checkpoint by using either:

  #. ``--checkpoint_dir <dir>`` Directory where the model checkpoint(s) are saved (i.e. ``--save_dir`` during training). This will walk the directory, load all ``.pt`` files it finds, and treat the models as an ensemble.
  #. ``--checkpoint_path <path>`` Path to a model checkpoint file (``.pt`` file).

* ``--preds_path`` Path where a CSV file containing the predictions will be saved.

For example:

.. code-block::

    chemprop predict --test_path data/tox21.csv \
        --checkpoint_dir tox21_checkpoints \
        --preds_path tox21_preds.csv

or

.. code-block::

    chemprop predict --test_path data/tox21.csv \
        --checkpoint_path tox21_checkpoints/fold_0/model_0/model.pt \
        --preds_path tox21_preds.csv
