Command Line Tutorial
=====================

Chemprop may be invoked from the command line using the following command:

.. code-block::

    $ chemprop COMMAND [ARGS]

where ``COMMAND`` is one of the following:

* ``train``: Train a model.
* ``predict``: Make predictions with a trained model.
* ``hyperopt``: Perform hyperparameter optimization.
* ``interpret``: Interpret model predictions.

and ``ARGS`` are command-specific arguments. To see the arguments for a specific command, run:

.. code-block::

    $ chemprop COMMAND --help

For example, to see the arguments for the ``train`` command, run:

.. code-block::

    $ chemprop train --help

For more details on each command, see the corresponding section below:

* :ref:`train`
* :ref:`predict`
* :ref:`interpret`

.. toctree::
    :maxdepth: 1
    :hidden:

    train
    predict
    interpret
