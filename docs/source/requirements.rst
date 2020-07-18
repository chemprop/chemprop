.. _requirements:

Requirements
============

For small datasets (~1000 molecules), it is possible to train models within a few minutes on a standard laptop with CPUs only. However, for larger datasets and larger chemprop models, we recommend using a GPU for significantly faster training.

To use chemprop with GPUs, you will need:

* cuda >= 8.0
* cuDNN

Chemprop is uses Python 3.6+ and all models are built with `PyTorch <https://pytorch.org/>`_. See :ref:`installation` for details on how to install Chemprop and its dependencies.

