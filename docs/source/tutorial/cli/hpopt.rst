.. _hpopt:

Hyperparameter Optimization
============================

.. note::
   Chemprop relies on `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ for hyperparameter optimization, which is not yet compatible with python=3.12 and is an optional install. To install the required dependencies, run :code:`pip install 'chemprop[hpopt]'` if installing with Option 1, or :code:`pip install -e ".[hpopt]"` if installing with Option 2 or 3.

Searching Hyperparameter Space
--------------------------------

We include an automated hyperparameter optimization procedure through the Ray Tune package. Hyperparameter optimization can be run as follows:

.. code-block::

   chemprop hpopt --data-path <data_path> --task-type <task> --search-parameter-keywords <keywords> --hpopt-save-dir <save_dir>

For example:

.. code-block::

   chemprop hpopt --data-path tests/data/regression.csv \
   --task-type regression \
   --search-parameter-keywords depth ffn_num_layers message_hidden_dim \
   --hpopt-save-dir results 

The search parameters can be any combination of hyperparameters or a predefined set. Options include :code:`basic` (default), which consists of:

 * :code:`depth` 
 * :code:`ffn_num_layers`
 * :code:`dropout` 
 * :code:`message_hidden_dim`
 * :code:`ffn_hidden_dim`

Another option is :code:`learning_rate` which includes:

 * :code:`max_lr`
 * :code:`init_lr_ratio`
 * :code:`final_lr_ratio`
 * :code:`warmup_epochs`

Other individual search parameters include:

 * :code:`activation`
 * :code:`aggregation`
 * :code:`aggregation_norm`
 * :code:`batch_size`

Specifying :code:`--search-parameter-keywords all` will search over all 13 of the above parameters.

The following other common keywords may be used:

 * :code:`--raytune-num-samples <num_samples>` The number of conditions to sample 
 * :code:`--hyperopt-random-state-seed <seed>` The random state seed used during hyperparameter optimization.

Other keywords related to Raytune training are also available (see :ref:`cmd` for a full list).

Applying Optimal Hyperparameters
---------------------------------

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path. For example:

.. code-block::

   chemprop train --data-path tests/data/regression.csv \
   --task-type regression \
   --config-path results/best_config.toml \

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. 

.. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run :code:`chemprop hpopt` separately on each split's training and validation data with test held out.
