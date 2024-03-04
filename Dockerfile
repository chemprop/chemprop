# Dockerfile
#
# Builds a Docker image containing Chemprop and its required dependencies.
#
# Build this image with:
#  docker build .
#
# Run the built image with:
#  docker run --name chemprop_container -it <IMAGE_ID>
# where <IMAGE_ID> is shown from the output of the build command.
#
# Note:
# This image only runs on CPU - we do not provide a GPU Dockerfile
# because it is highly system-dependent and much harder than just installing
# from source or manually installing inside a Docker container.
# 
# To disregard this advice and make this Dockerfilework for your GPU,
# select a new parent image from:
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
# and then update the installation steps to install the appropriate
# versions of PyTorch:
# https://pytorch.org/get-started/locally/
# and PyTorch Scatter:
# https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file#installation
# based on your GPU version.
#
# We have absolutely no idea how to make the above work on AMD GPUs... ¯\_(ツ)_/¯
# Good luck!

# Parent Image
FROM ubuntu:latest

# 'Install' Bash shell
RUN ln -snf /bin/bash /bin/sh

# Install system dependencies
#
# List of deps and why they are needed:
#  - git for downloading repository
#  - wget for downloading conda install script
#  - libxrender1 required by RDKit
RUN apt-get update && \
    apt-get install -y \
    wget \
    git \
    libxrender1 && \
    apt-get autoremove -y && \
    apt-get clean -y

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="$PATH:/miniconda/bin"

# Set Bash as the default shell for following commands
SHELL ["/bin/bash", "-c"]

WORKDIR /opt/chemprop

# build the conda environment
RUN conda create --name chemprop_env python=3.11* && \
    conda clean --all --yes

# This runs all subsequent commands inside the chemprop_env conda environment
#
# Analogous to just activating the environment, which we can't actually do here
# since that requires running conda init and restarting the shell (not possible
# in a Dockerfile build script)
SHELL ["conda", "run", "--no-capture-output", "-n", "chemprop_env", "/bin/bash", "-c"]

# Follow the installation instructions (but with fixed versions, for stability of this image) then clear the cache
RUN python -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    python -m pip install git+https://github.com/chemprop/chemprop.git@f39a672d003dd82f1fddff8009d98a0c4f21796b && \
    python -m pip cache purge

# when running this image, open an interactive bash terminal inside the conda environment
RUN echo "source activate chemprop_env" > ~/.bashrc
ENTRYPOINT ["/bin/bash", "--login"]
