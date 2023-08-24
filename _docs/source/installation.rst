.. _installation:

Installation
============

Overview
--------

Chemprop can either be installed from PyPi via pip or from source (i.e., directly from the git repo). The PyPi version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source.

Conda
-----

Both options require conda, so first install Miniconda from `<https://conda.io/miniconda.html>`_.

Then proceed to either option below to complete the installation. Note that on machines with GPUs, you may need to manually install a GPU-enabled version of PyTorch by following the instructions `here <https://pytorch.org/get-started/locally/>`_.

Option 1: Installing from PyPi
------------------------------

1. :code:`conda create -n chemprop python=3.8`
2. :code:`conda activate chemprop`
3. :code:`conda install -c conda-forge rdkit`
4. :code:`pip install git+https://github.com/bp-kelley/descriptastorus`
5. :code:`pip install chemprop`

Option 2: Installing from source
--------------------------------

1. :code:`git clone https://github.com/chemprop/chemprop.git`
2. :code:`cd chemprop`
3. :code:`conda env create -f environment.yml`
4. :code:`conda activate chemprop`
5. :code:`pip install -e .`

Docker
------

Chemprop can also be installed with Docker. Docker makes it possible to isolate the Chemprop code and environment. To install and run our code in a Docker container, follow these steps:

1. :code:`git clone https://github.com/chemprop/chemprop.git`
2. :code:`cd chemprop`
3. Install Docker from `<https://docs.docker.com/install/>`_
4. :code:`docker build -t chemprop .`
5. :code:`docker run -it chemprop:latest`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs.
Alternatively, with Docker 19.03+, you can specify the :code:`--gpus` command line option instead.

In addition, you will also need to ensure that the CUDA toolkit version in the Docker image is compatible with the CUDA driver on your host machine.
Newer CUDA driver versions are backward-compatible with older CUDA toolkit versions.
To set a specific CUDA toolkit version, add :code:`cudatoolkit=X.Y` to :code:`environment.yml` before building the Docker image.
