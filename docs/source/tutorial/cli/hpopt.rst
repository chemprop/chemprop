.. _hpopt:

.. warning:: 
    This page is under construction.


Hyperparameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Although the default message passing architecture works quite well on a variety of datasets, optimizing the hyperparameters for a particular dataset often leads to marked improvement in predictive performance. We have automated hyperparameter optimization via Bayesian optimization (using the `hyperopt <https://github.com/hyperopt/hyperopt>`_ package), which will find the optimal hidden size, depth, dropout, and number of feed-forward layers for our model. Optimization can be run as follows:

.. code-block::

   chemprop hpopt --data_path <data_path> --task-type <type> --num_iters <n> --config_save_path <config_path>

where :code:`<n>` is the number of hyperparameter settings to try and :code:`<config_path>` is the path to a :code:`.json` file where the optimal hyperparameters will be saved.

If installed from source, :code:`chemprop hpopt` can be replaced with :code:`python hyperparameter_optimization.py`.

Once hyperparameter optimization is complete, the optimal hyperparameters can be applied during training by specifying the config path as follows:

.. code-block::

   chemprop train --data_path <data_path> --task-type <type> --config_path <config_path>

Note that the hyperparameter optimization script sees all the data given to it. The intended use is to run the hyperparameter optimization script on a dataset with the eventual test set held out. If you need to optimize hyperparameters separately for several different cross validation splits, you should e.g. set up a bash script to run hyperparameter_optimization.py separately on each split's training and validation data with test held out.
