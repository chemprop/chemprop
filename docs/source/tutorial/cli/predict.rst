.. _predict:

Prediction
----------

To load a trained model and make predictions, run:

.. code-block::
   
    chemprop predict --test-path <test_path> --model-path <model_path>

where :code:`<test_path>` is the path to the data to test on, and :code:`<model_path>` is the location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, will recursively search and predict on all found (.pt) models. By default, predictions will be saved to the same directory as the test path. If desired, a different directory can be specified by using :code:`--preds-path <path>`. The predictions <path> can end with either .csv or .pkl, and the output will be saved to the corresponding file type.

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
