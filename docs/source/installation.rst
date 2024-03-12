.. _installation:

Installation
============

Chemprop can either be installed from PyPI via pip_ or from source (i.e., directly from the `git repo`_). The PyPI version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source. We recommend installing ``chemprop`` in a virtual environment (e.g., conda_ or miniconda_). The following sections assume you are using ``conda`` or ``miniconda``, but you can use any virtual environment manager you like.

.. _pip: https://pypi.org/project/chemprop/
.. _git repo: https://github.com/chemprop/chemprop.git
.. _conda: https://docs.conda.io/en/latest/conda.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. note:: 
    We also plan to make chemprop installable using `Docker` or using an `environment.yml` file with `conda` before the release of v2.0.0.

Start by setting up your virtual environment. We assume you are using ``conda`` or ``miniconda``, but you may adapt these steps use any virtual environment manager you like:

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop

.. note:: 
    *CPU-only installation:* If you do not have a GPU, install a CPU-only version of PyTorch by running the following command before installing Chemprop:

    .. code-block::

        conda install pytorch cpuonly -c pytorch

.. note:: 
    We are aware that some users may experience issues during installation while trying to install `torch-scatter`. This is an issue with the `torch-scatter` package and not with Chemprop. We will resolve this issue before the release of v2.0.0, most likely by replacing our `torch-scatter` functions with native PyTorch functions and removing the `torch-scatter` dependency. You can follow along with this issue here: https://github.com/chemprop/chemprop/issues/580.

Option 1: Installing from PyPI
------------------------------

.. code-block::

    pip install torch
    pip install torch-scatter
    pip install chemprop --pre


Option 2: Installing from source
--------------------------------

.. code-block::

    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    git checkout v2/dev
    pip install torch
    pip install torch-scatter
    pip install .

.. Option 3: Installing via Docker
.. -------------------------------

.. Chemprop can also be installed with Docker, making it possible to isolate the Chemprop code and environment. To install and run our code in a Docker container, first install docker from docker_. Then, run the following commands:

.. .. code-block::

..     git clone https://github.com/chemprop/chemprop.git
..     cd chemprop
..     git checkout v2/dev
..     docker build --tag chemprop . --build-arg="CUDA=<cuda_arg>"
..     docker run -it chemprop:latest


.. .. note:: 
..     In the docker build line, replace ``<cuda_arg>`` with ``cpu``, ``cu118``, or ``cu121`` depending on your version of PyTorch. If experiencing permission errors, prepend ``sudo`` to the Docker commands.

..     You will need to run the last command with ``nvidia-docker`` if you are on a GPU machine in order to be able to access the GPUs. Alternatively, with ``docker >= 19.03``, you can specify the ``--gpus`` command line option instead.

..     In addition, you will also need to ensure that the CUDA toolkit version in the Docker image is compatible with the CUDA driver on your host machine. Newer CUDA driver versions are backward-compatible with older CUDA toolkit versions. To set a specific CUDA toolkit version, add ``cudatoolkit=X.Y`` to ``environment.yml`` before building the Docker image.

