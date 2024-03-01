FROM mambaorg/micromamba:0.23.0
ARG CUDA=cpu

USER root

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/chemprop

WORKDIR /opt/chemprop

RUN /opt/conda/bin/python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+${CUDA}.html && \
    /opt/conda/bin/python -m pip install torch && \
    /opt/conda/bin/python -m pip install --upgrade-strategy only-if-needed -e .

