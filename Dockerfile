FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN echo "source activate root" > ~/.bashrc
RUN conda env update -n root --file environment.yml
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install tensorflow-gpu
# use pip install tensorflow for cpu

