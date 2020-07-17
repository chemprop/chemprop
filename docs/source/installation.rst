Installation
============

Requirements
------------

For small datasets (~1000 molecules), it is possible to train models within a few minutes on a standard laptop with CPUs only. However, for larger datasets and larger chemprop models, we recommend using a GPU for significantly faster training.

To use chemprop with GPUs, you will need:

* cuda >= 8.0
* cuDNN

Installing Chemprop
-------------------

Chemprop can either be installed from PyPi via pip or from source (i.e., directly from this git repo). The PyPi version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source.

Both options require conda, so first set up a conda environment and install RDKit:

1. Install Miniconda from `<https://conda.io/miniconda.html>`_
2. :code:`conda env create -n chemprop python=3.7`
3. :code:`conda activate chemprop`
4. :code:`conda install -c conda-forge rdkit`

Then proceed to either option below to complete the installation. Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions `here <https://pytorch.org/get-started/locally/>`_.

Option 1: Installing from PyPi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Coming soon**

1. :code:`pip install chemprop`

Option 2: Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. :code:`git clone https://github.com/chemprop/chemprop.git`
2. :code:`cd chemprop`
3. :code:`pip install -e .`

Docker
^^^^^^

Chemprop can also be installed with Docker. Docker makes it possible to isolate the Chemprop code and environment. To install and run our code in a Docker container, follow these steps:

1. :code:`git clone https://github.com/chemprop/chemprop.git`
2. :code:`cd chemprop`
3. Install Docker from `<https://docs.docker.com/install/>`_
4. :code:`docker build -t chemprop .`
5. :code:`docker run -it chemprop:latest /bin/bash`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs.
