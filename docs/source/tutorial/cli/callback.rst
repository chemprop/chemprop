.. _callback:

Custom Callbacks for Prediction
===============================

Chemprop's :code:`predict` subcommand can be invoked with the :code:`--callback <cb>` flag, allowing for the execution of custom code during the prediction process.

Interpretability Callbacks
--------------------------

Myerson Values
^^^^^^^^^^^^^^

A Myerson explainer calculates and saves Myerson explanations during a :code:`predict` call using the `myerson <https://myerson.readthedocs.io/en/latest/>`_ package.
It can be invoked by passing :code:`--callback myerson` to the :code:`chemprop predict` command.

The explanations take the form of node attributions and are saved as a pickle file containing a dictionary with the keys :code:`myerson_values` and :code:`sampled`.
The :code:`myerson_values` will be a list of 1D or 2D arrays of shape :code:`num_atoms` (for regression or binary classification)
or :code:`num_atoms x num_classes` (for multilabel binary classification) containing the explanations.

By default, molecules with more than 20 nodes will use a sampling explainer instead of the exact explainer.
This threshold is controlled by the :code:`sampling_threshold` parameter (currently hardcoded to 20 when using the CLI).

Myerson Explanations are not supported for atom or bond level predictions. Furthermore, they are only implemented for models using a :code:`BinaryClassificationFFN` or :code:`RegressionFFN`.

.. code-block::
   
    chemprop predict --test-path tests/data/smis.csv --model-paths tests/data/example_model_v2_regression_mol.ckpt --callback myerson
