.. _installation:

Installation
============

Chemprop can either be installed from PyPI via pip_ or using a provided ``x.y.x_requirements.txt`` file, from source (i.e., directly from the `git repo`_) using ``pip`` or the ``environment.yml`` file, or from `Docker`_. The PyPI version includes the vast majority of Chemprop functionality, but some functionality is only accessible when installed from source. We recommend installing ``chemprop`` in a virtual environment (e.g., conda_ or miniconda_). The following sections assume you are using ``conda`` or ``miniconda``, but you can use any virtual environment manager you like (e.g. ``mamba``).

.. _pip: https://pypi.org/project/chemprop/
.. _git repo: https://github.com/chemprop/chemprop.git
.. _`Docker`: https://www.docker.com/get-started/
.. _conda: https://docs.conda.io/en/latest/conda.html
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

.. note::
    *Python 3.11 vs. 3.12:* Options 1, 2, and 4 below explicitly specify ``python=3.11`` but you can choose to replace ``python=3.11`` with ``python=3.12`` in these commands. We test Chemprop on both versions in our CI.

.. note:: 
    *CPU-only installation:* For the following options 1-3, if you do not have a GPU, you might need to manually install a CPU-only version of PyTorch. This should be handled automatically, but if you find that it is not, you should run the following command before installing Chemprop:

    .. code-block::

        conda install pytorch cpuonly -c pytorch

.. _install-from-pypi:

Option 1: Installing from PyPI
------------------------------

To install the latest version of Chemprop and all of its dependencies, execute the following commands:

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop
    pip install chemprop

The above approach is recommended as it will install the most updated versions of all external dependencies and will be more compatible with other dependencies you may wish to add. However, it is possible that external dependencies to Chemprop may introduce backwards incompatible changes to their codebase. For this reason, we also provide known working sets of dependencies for each Chemprop version. To install a specific version of Chemprop (i.e. 2.1.0), download the corresponding ``x.y.z_requirements.txt`` file from the `Chemprop GitHub repository <https://github.com/chemprop/chemprop/tree/main/requirements>`_ and run the following commands:

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop
    pip install -r x.y.z_requirements.txt

.. _install-from-source-using-pip:

Option 2: Installing from source using pip
------------------------------------------

.. code-block::

    conda create -n chemprop python=3.11
    conda activate chemprop
    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    pip install -e .

.. note:: 
    You can also use this option to install additional optional dependencies by replacing ``pip install -e .`` with ``pip install -e ".[hpopt,dev,docs,test,notebooks]"``.

.. _install-from-source-using-environment-yml:

Option 3: Installing from source using environment.yml
-------------------------------------------------------

.. code-block::

    git clone https://github.com/chemprop/chemprop.git
    cd chemprop
    conda env create -f environment.yml
    conda activate chemprop
    pip install -e .

.. note::
    `cuik-molmaker`_ is a package that accelerates featurization of molecules using RDKit, and thereby accelerates training and inference. It can be installed using the python script ``check_and_install_cuik_molmaker.py``. This script finds a compatible version of ``cuik-molmaker`` depending on the version of ``RDKit`` and ``PyTorch`` and installs it. Currently, ``cuik-molmaker`` is compatible with installation :ref:`install-from-source-using-environment-yml` and :ref:`install-via-docker`.

.. _`cuik-molmaker`: https://github.com/NVIDIA-Digital-Bio/cuik-molmaker

.. _install-via-docker:

Option 4: Installing via Docker
-------------------------------

Chemprop can also be installed with Docker, making it possible to isolate the Chemprop code and environment.
To install and run Chemprop in a Docker container, first `install Docker`_.
You may then either ``pull`` and use official Chemprop images or ``build`` the image yourself.

.. _`install Docker`: https://docs.docker.com/get-docker/

.. note:: 
    The Chemprop Dockerfile runs only on CPU and does not support GPU acceleration.
    Linux users with NVIDIA GPUs may install the `nvidia-container-toolkit`_ from NVIDIA and modify the installation instructions in the Dockerfile to install the version of `torch` which is compatible with your system's GPUs and drivers.
    Adding the ``--gpus all`` argument to ``docker run`` will then allow Chemprop to run on GPU from within the container. You can see other options for exposing GPUs in the `Docker documentation`_.
    Users on other systems should install Chemprop from PyPI or source.

.. _`nvidia-container-toolkit`: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
.. _`Docker documentation`: https://docs.docker.com/config/containers/resource_constraints/#expose-gpus-for-use

Pull Official Images
++++++++++++++++++++

.. code-block::

    docker pull chemprop/chemprop:X.Y.Z
    docker run -it chemprop/chemprop:X.Y.Z

Where ``X``, ``Y``, and ``Z`` should be replaced with the version of Chemprop you wish to ``pull``.
For example, to pull ``chemprop-2.0.0`` run

.. code-block::

    docker pull chemprop/chemprop:2.0.0

.. note::
    Not all versions of Chemprop are available as pre-built images.
    Visit the `Docker Hub`_ page for a list of those that are available.

.. note::
    Nightly builds of Chemprop are available under the ``latest`` tag on Dockerhub and are intended for developer use and as feature previews, not production deployment.

.. _`Docker Hub`: https://hub.docker.com/repository/docker/chemprop/chemprop/general

Build Image Locally
+++++++++++++++++++

See the build instructions in the top of the ``Dockerfile``.
