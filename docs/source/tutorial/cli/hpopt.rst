.. _hpopt:

Hyperparameter Optimization
============================

.. note::
    Chemprop relies on `Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ for hyperparameter optimization, which is not yet compatible with python=3.12 and is an optional install. To install the required dependencies, make sure your Python version is 3.11 and run :code:`pip install -U ray[tune]` if installing with PyPI, or :code:`pip install -e .[hpopt]` if installing from source.

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

 * :code:`depth` The number of message passing steps
 * :code:`ffn_num_layers` The number of layers in the FFN model
 * :code:`dropout` The probability (from 0.0 to 1.0) of dropout in the MPNN & FNN layers
 * :code:`message_hidden_dim` The hidden dimension in the message passing step 
 * :code:`ffn_hidden_dim` The hidden dimension in the FFN model

Another option is :code:`learning_rate` which includes:

 * :code:`max_lr` The maximum learning rate
 * :code:`init_lr` The initial learning rate. It is searched as a ratio relative to the max learning rate
 * :code:`final_lr` The initial learning rate. It is searched as a ratio relative to the max learning rate 
 * :code:`warmup_epochs` Number of warmup epochs, during which the learning rate linearly increases from the initial to the maximum learning rate

Other individual search parameters include:

 * :code:`activation` The activation function used in the MPNN & FFN layers. Choices include ``relu``, ``leakyrelu``, ``prelu``, ``tanh``, ``selu``, and ``elu``
 * :code:`aggregation` Aggregation mode used during molecule-level predictor. Choices include ``mean``, ``sum``, ``norm``
 * :code:`aggregation_norm` For ``norm`` aggregation, the normalization factor by which atomic features are divided
 * :code:`batch_size` Batch size for dataloader

Specifying :code:`--search-parameter-keywords all` will search over all 13 of the above parameters.

The following other common keywords may be used:
 
 * :code:`--raytune-num-samples <num_samples>` The number of trials to perform
 * :code:`--raytune-num-cpus <num_cpus>` The number of CPUs to use  
 * :code:`--raytune-num-gpus <num_gpus>` The number of GPUs to use  
 * :code:`--raytune-max-concurrent-trials <num_trials>` The maximum number of concurrent trials
 * :code:`--raytune-search-algorithm <algorithm>` The choice of control search algorithm (either ``random``, ``hyperopt``, or ``optuna``). If ``hyperopt`` is specified, then the arguments ``--hyperopt-n-initial-points <num_points>`` and ``--hyperopt-random-state-seed <seed>`` can be specified.

Other keywords related to hyperparameter optimization are also available (see :ref:`cmd` for a full list).

Splitting
----------
By default, Chemprop will split the data into train / validation / test data splits. The splitting behavior can be modified using the same splitting arguments used in training, i.e., section :ref:`train_validation_test_splits`.

.. note::
    This default splitting behavior is different from Chemprop v1, wherein the hyperparameter optimization was performed on the entirety of the data provided to it.

If ``--num-folds`` is greater than one, Chemprop will only use the first split to perform hyperparameter optimization. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run :code:`chemprop hpopt` separately on each split.


Applying Optimal Hyperparameters
---------------------------------

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path. If an argument is both provided via the command line and the config file, the command line takes precedence. For example:

.. code-block::

    chemprop train --data-path tests/data/regression.csv \
        --config-path results/best_config.toml
