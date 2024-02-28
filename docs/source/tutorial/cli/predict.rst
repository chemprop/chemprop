.. _predict:

Prediction
----------

To load a trained model and make predictions, run ``chemprop predict`` and specify:

 * :code:`--test-path <path>` Path to the data to predict on.
 * :code:`--model-path <path>` Path to the trained model.

By default, predictions will be saved to the same directory as the test path. If desired, a different directory can be specified by using :code:`--preds-path <path>`

For example:

.. code-block::
  
    chemprop predict --test-path data/tox21.csv \
        --model-path tox21/model_0/model.pt \
        --preds-path tox21_preds.csv

Specifying Data to Parse
^^^^^^^^^^^^^^^^^^^^^^^^

By default, Chemprop will assume that the the 0th column in the data .csv will have the data. To use a separate column, specify:

 * :code:`--smiles-columns` Text label of the column that includes the SMILES strings

If atom-mapped reaction SMILES are desired, specify:

 * :code:`--reaction-columns` Text labels of the columns that include the reaction SMILES

If :code:`--reaction-mode` was specified during training, those same flags must be specified for the prediction step.