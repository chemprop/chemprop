# Dockerfile
#
# Builds a Docker image containing Chemprop and its required dependencies.
#
# Build this image with:
#  git clone https://github.com/chemprop/chemprop.git
#  cd chemprop
#  docker build --tag=chemprop:latest .
#
# Run the built image with:
#  docker run --name chemprop_container -it chemprop:latest
#
# Note:
# This image only runs on CPU - we do not provide a Dockerfile
# for GPU use (see installation documentation). 

# Parent Image
FROM condaforge/miniforge3:latest

# Install libxrender1 (required by RDKit) and binutils (to get strings, for cuik-molmaker install) and then clean up
RUN apt-get update && \
    apt-get install -y \
    libxrender1 binutils && \
    apt-get autoremove -y && \
    apt-get clean -y

WORKDIR /opt/chemprop

# build an empty conda environment with appropriate Python version
RUN conda create --name chemprop_env python=3.11* rdkit=2025.3.2 pytorch=2.6.0 -c pytorch

# This runs all subsequent commands inside the chemprop_env conda environment
#
# Analogous to just activating the environment, which we can't actually do here
# since that requires running conda init and restarting the shell (not possible
# in a Dockerfile build script)
SHELL ["conda", "run", "--no-capture-output", "-n", "chemprop_env", "/bin/bash", "-c"]

# Follow the installation instructions then clear the cache
ADD chemprop chemprop
ENV PYTHONPATH=/opt/chemprop
ADD LICENSE.txt pyproject.toml README.md environment.yml ./
RUN conda env update --file environment.yml --name chemprop_env && \
    conda clean --all --yes && \
    python -m pip install --no-deps .

# Install cuik-molmaker using script
RUN python /opt/chemprop/chemprop/scripts/check_and_install_cuik_molmaker.py

# when running this image, open an interactive bash terminal inside the conda environment
RUN conda init
RUN echo "conda activate chemprop_env" >> ~/.bashrc

ENTRYPOINT ["/bin/bash", "--login"]
