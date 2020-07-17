Training
========


To train a model, run:

.. code-block::

   chemprop_train --data_path <path> --dataset_type <type> --save_dir <dir>

where :code:`<path>` is the path to a CSV file containing a dataset, :code:`<type>` is either "classification" or "regression" depending on the type of the dataset, and :code:`<dir>` is the directory where model checkpoints will be saved.

For example:

.. code-block::

   chemprop_train --data_path data/tox21.csv --dataset_type classification --save_dir tox21_checkpoints

A full list of available command-line arguments can be found in :ref:`args`.

If installed from source, :code:`chemprop_train` can be replaced with :code:`python train.py`.

Notes:
* The default metric for classification is AUC and the default metric for regression is RMSE. Other metrics may be specified with :code:`--metric <metric>`.
* :code:`--save_dir` may be left out if you don't want to save model checkpoints.
* :code:`--quiet` can be added to reduce the amount of debugging information printed to the console. Both a quiet and verbose version of the logs are saved in the :code:`save_dir`.
