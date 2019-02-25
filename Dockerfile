FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# conda install code from https://hub.docker.com/r/kundajelab/cuda-anaconda-base/dockerfile, modified for python3

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 \
git mercurial subversion libbz2-dev libz-dev libpng-dev

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /app
COPY . /app

RUN conda install pip
RUN conda install -c rdkit nox
RUN conda install cairo
RUN conda env update -n base --file environment.yml
RUN pip install git+https://github.com/bp-kelley/descriptastorus
