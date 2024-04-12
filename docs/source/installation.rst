.. _installation:

Installation
============

Chemprop can either be installed from PyPI via pip_, from source (i.e., directly from the `git repo`_), or from `Docker`_. The PyPI version includes a vast majority of Chemprop functionality, but some functionality is only accessible when installed from source. We recommend installing ``chemprop`` in a virtual environment (e.g., conda_ or miniconda_). The following sections assume you are using ``conda`` or ``miniconda``, but you can use any virtual environment manager you like.

.. _pip: https://pypi.org/project/chemprop/
.. _git repo: https://github.com/chemprop/chemprop.git
.. _`Docker`: https://www.docker.com/get-started/
.. _conda: https://docs.conda.io/en/latest/conda.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. note:: 
    We also plan to make chemprop installable using an ``environment.yml`` file with ``conda`` before the release of v2.0.0.

Start by setting up your virtual environment. We assume you are using ``conda`` or ``miniconda``, but you may adapt these steps use any virtual environment manager you like:

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop

.. note:: 
    *CPU-only installation:* If you do not have a GPU, install a CPU-only version of PyTorch by running the following command before installing Chemprop:

    .. code-block::

        conda install pytorch cpuonly -c pytorch

Option 1: Installing from PyPI
------------------------------

.. code-block::

    pip install chemprop --pre


Option 2: Installing from source
--------------------------------

.. code-block::

    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    pip install .

Option 3: Installing via Docker
-------------------------------
 
Chemprop can also be installed with Docker, making it possible to isolate the Chemprop code and environment.
To install and run Chemprop in a Docker container, first install `Docker`_.
You may then either ``pull`` and use official Chemprop images or ``build`` the image yourself.

.. note:: 
    The Chemprop Dockerfile runs only on CPU and does not support GPU acceleration.
    Linux users with NVIDIA GPUs may install the `nvidia-container-toolkit`_ from NVIDIA and modify the installation instructions in the Dockerfile to install versions of `torch` and `torch-scatter` which are compatible with your system's GPUs and drivers.
    Adding the ``--gpus all`` argument to ``docker run`` will then allow Chemprop to run on GPU from within the container.
    Users on other systems should install Chemprop from PyPI or source.

.. _`nvidia-container-toolkit`: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Pull Official Images
++++++++++++++++++++

.. code-block::

    docker pull chemprop/chemprop:X.Y.ZrcN
    docker run -it chemprop/chemprop:X.Y.ZrcN

Where ``X``, ``Y``, ``Z``, and ``N``, should be replaced with the version of Chemprop you wish to ``pull``.
For example, to pull ``chemprop-2.0.0rc1`` run

.. code-block::

    docker pull chemprop/chemprop:2.0.0rc1

Note that not all versions of Chemprop are available as pre-built images.
Visit the `Docker Hub`_ page for a list of those that are available.

.. _`Docker Hub`: https://hub.docker.com/repository/docker/chemprop/chemprop/general

Build Image Locally
+++++++++++++++++++

First follow the instructions in `Option 2: Installing from Source`_ up to invoking ``pip``, and then run the following:

.. code-block::

    docker build --tag=chemprop .
    docker run -it chemprop
