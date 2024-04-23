.. _tutorial:

Command Line Tutorial
=====================

Chemprop may be invoked from the command line using the following command:

.. code-block::

    $ chemprop COMMAND [ARGS]

where ``COMMAND`` is one of the following:

* ``train``: Train a model.
* ``predict``: Make predictions with a trained model.
* ``convert``: Convert a trained Chemprop model from v1 to v2.

and ``ARGS`` are command-specific arguments. To see the arguments for a specific command, run:

.. code-block::

    $ chemprop COMMAND --help

For example, to see the arguments for the ``train`` command, run:

.. code-block::

    $ chemprop train --help

To enable logging, specify ``--log <path/to/logfile>`` or ``--logfile <path/to/logfile>``, where ``<path/to/logfile>`` is the desired path to which the logfile should be written; if unspecified, the log will be written to ``chemprop_logs``.
If more detailed debugging information is required, specify ``-v``, ``-vv``, or ``-vvv`` (in increasing order of detail).

Chemprop is built on top of Lightning, which has support for training and predicting on GPUs.
Relevant CLI flags include `--accelerator` and `--devices`.
See the `Lightning documentation <https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html#choosing-gpu-devices>`_ and CLI reference for more details.

For more details on each command, see the corresponding section below:

* :ref:`train`
* :ref:`predict`
* :ref:`convert`

The following features are not yet implemented, but will soon be included in a future release:

* ``hyperopt``: Perform hyperparameter optimization.
* ``interpret``: Interpret model predictions.
* ``fingerprint``: Use a trained model to compute a learned representation.


.. toctree::
    :maxdepth: 1
    :hidden:

    train
    predict
    convert