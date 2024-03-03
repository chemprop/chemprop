.. _predict:

Prediction
----------

To load a trained model and make predictions, run:

.. code-block::
   
   chemprop predict --test-path <test_path> --model-path <model_path>

where :code:`<test_path>` is the path to the data to test on, and :code:`<model_path>` is the path to the trained model. By default, predictions will be saved to the same directory as the test path. If desired, a different directory can be specified by using :code:`--preds-path <path>`

For example:

.. code-block::
  
    chemprop predict --test-path tests/data/smis.csv \
        --model-path tests/data/example_model_v2_regression_mol.ckpt \
        --preds-path preds.csv


Specifying Data to Parse
^^^^^^^^^^^^^^^^^^^^^^^^

By default, Chemprop will assume that the the 0th column in the data .csv will have the data. To use a separate column, specify:

 * :code:`--smiles-columns` Text label of the column that includes the SMILES strings

If atom-mapped reaction SMILES are used, specify:

 * :code:`--reaction-columns` Text labels of the columns that include the reaction SMILES

If :code:`--reaction-mode` was specified during training, those same flags must be specified for the prediction step.
