.. _fingerprint:

Fingerprint
============================

To calculate the learned representations (encodings) of model inputs from a pretrained model, run

.. code-block::
   
    chemprop fingerprint --test-path <test_path> --model-path <model_path> 

where :code:`<test_path>` is the path to the CSV file containing SMILES strings, and :code:`<model_path>` is the location of checkpoint(s) or model file(s) to use for prediction. It can be a path to either a single pretrained model checkpoint (.ckpt) or single pretrained model file (.pt), a directory that contains these files, or a list of path(s) and directory(s). If a directory, will recursively search and predict on all found (.pt) models. By default, predictions will be saved to the same directory as the test path. If desired, a different directory can be specified by using :code:`--output <path>`. The output <path> can end with either .csv or .npz, and the output will be saved to the corresponding file type.

For example:

.. code-block::
  
    chemprop fingerprint --test-path tests/data/smis.csv \
        --model-path tests/data/example_model_v2_regression_mol.ckpt \
        --output fps.csv


Specifying FFN encoding layer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, the encodings are returned from the penultimate linear layer of the model's FFN. However, the exact layer to draw encodings from can be specified using :code:`--ffn-block-index <index>`.

An index of 0 will simply return the post-aggregation representation without passing through the FFN. Here, an index of 1 will return the output of the first linear layer of the FFN, an index of 2 the second layer, and so on.


Specifying Data to Parse
^^^^^^^^^^^^^^^^^^^^^^^^

:code:`fingerprint` shares the same arguments for specifying SMILES columns and reaction types as :code:`predict`. For more detail, see :ref:`predict`.