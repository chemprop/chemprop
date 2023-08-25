.. _installation:

Installation
============

Chemprop can either be installed from PyPi via pip_, from source (i.e., directly from the `git repo`_), or from docker_. The PyPi version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source. We recommend installing ``chemprop`` in a virtual environment (e.g., conda_ or miniconda_). The following sections assume you are using ``conda`` or ``miniconda``, but you can use any virtual environment manager you like.

.. _pip: https://pypi.org/project/chemprop/
.. _git repo: https://github.com/chemprop/chemprop.git
.. _docker: https://docker.com
.. _conda: https://docs.conda.io/en/latest/conda.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

Start by setting up your virtual environment. We assume you are using ``conda`` or ``miniconda``, but you may adapt these steps use any virtual environment manager you like:

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop

*CPU-only installation:*

If you do not have a GPU, install a CPU-only version of PyTorch by running the following command before installing Chemprop:

.. code-block::

    conda install pytorch cpuonly -c pytorch

**Option 1:** Installing from PyPI

.. code-block::

    pip install chemprop

**Option 2:** Installing from source

.. code-block::

    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    pip install .

**Option 3:** Installing via Docker

Chemprop can also be installed with Docker, making it possible to isolate the Chemprop code and environment. To install and run our code in a Docker container, first install docker from docker_. Then, run the following commands:

.. code-block::

    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    docker build -t chemprop .
    docker run -it chemprop:latest

Note that you will need to run the latter command with ``nvidia-docker`` if you are on a GPU machine in order to be able to access the GPUs. Alternatively, with ``docker >= 19.03``, you can specify the ``--gpus`` command line option instead.

In addition, you will also need to ensure that the CUDA toolkit version in the Docker image is compatible with the CUDA driver on your host machine. Newer CUDA driver versions are backward-compatible with older CUDA toolkit versions. To set a specific CUDA toolkit version, add ``cudatoolkit=X.Y`` to ``environment.yml`` before building the Docker image.
