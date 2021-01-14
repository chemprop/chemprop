FROM continuumio/miniconda3:4.9.2

COPY environment.yml /tmp/environment.yml

RUN /opt/conda/bin/conda env update -n base --file /tmp/environment.yml && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    rm /tmp/environment.yml

COPY . /opt/chemprop

WORKDIR /opt/chemprop

RUN /opt/conda/bin/pip install -e .

ENV PATH /opt/conda/bin${PATH:+:${PATH}}

ENTRYPOINT ["/bin/bash"]
CMD ["-l"]
