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
# This image only runs on CPU - we do not provide a Dockerfile
# for GPU use (see installation documentation). 

# Parent Image
FROM continuumio/miniconda3:latest

# Install libxrender1 (required by RDKit) and then clean up
RUN apt-get update && \
    apt-get install -y \
    libxrender1 && \
    apt-get autoremove -y && \
    apt-get clean -y

WORKDIR /opt/chemprop

# build an empty conda environment with appropriate Python version
RUN conda create --name chemprop_env python=3.11* && \
    conda clean --all --yes

# This runs all subsequent commands inside the chemprop_env conda environment
#
# Analogous to just activating the environment, which we can't actually do here
# since that requires running conda init and restarting the shell (not possible
# in a Dockerfile build script)
SHELL ["conda", "run", "--no-capture-output", "-n", "chemprop_env", "/bin/bash", "-c"]

# Follow the installation instructions then clear the cache
ADD chemprop chemprop
ENV PYTHONPATH /opt/chemprop
ADD LICENSE.txt pyproject.toml README.md .
RUN python -m pip install . && \
    python -m pip cache purge

# when running this image, open an interactive bash terminal inside the conda environment
RUN echo "conda activate chemprop_env" > ~/.bashrc
ENTRYPOINT ["/bin/bash", "--login"]
