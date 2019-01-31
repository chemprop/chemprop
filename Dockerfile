FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN conda install -c rdkit nox
RUN conda install cairo
RUN conda env update -n base --file environment.yml
RUN pip install --trusted-host pypi.python.org -r requirements.txt
