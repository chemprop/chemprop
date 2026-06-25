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

The explanations take the form of node attributions and are saved as a compressed NumPy archive (:code:`.npz` file). Each molecule's explanation is saved as a separate array within the archive (e.g., :code:`arr_0`, :code:`arr_1`, etc.).
Each array will be a 1D or 2D NumPy array of shape :code:`num_atoms` (for regression or binary classification) or :code:`num_atoms x num_classes` (for multi-class classification) containing the explanation for one molecule. Alternatively, the explanations can be saved as a JSON file by setting the :code:`save_as_json` parameter to :code:`true`.
When saved as a JSON file, the output is a list of explanations, where each explanation corresponds to a molecule. For 2D explanations (multi-class), each inner list represents a column (i.e., attributions for a specific class across all atoms).

By default, molecules with more than 20 nodes will use a sampling explainer instead of the exact explainer.
This threshold is controlled by the :code:`sampling_threshold` parameter, which can be set using the :code:`--callback-params` flag, which expects a JSON string.

Myerson Explanations are not supported for atom or bond level predictions. Furthermore, they are only implemented for models using a :code:`BinaryClassificationFFN` or :code:`RegressionFFN`.

.. code-block::

    chemprop predict --test-path tests/data/smis.csv --model-paths tests/data/example_model_v2_regression_mol.ckpt --callback myerson
    chemprop predict --test-path tests/data/smis.csv --model-paths tests/data/example_model_v2_regression_mol.ckpt --callback myerson --callback-params '{"sampling_threshold": 25}'
    chemprop predict --test-path tests/data/smis.csv --model-paths tests/data/example_model_v2_regression_mol.ckpt --callback myerson --callback-params '{"save_as_json": true}'
