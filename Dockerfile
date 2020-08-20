FROM continuumio/miniconda:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ADD ./environment.yml /src/environment.yml
RUN conda env create -f /src/environment.yml

ENV PATH /opt/conda/envs/flatland-rl/bin:$PATH
ENV CONDA_DEFAULT_ENV flatland-rl
ENV CONDA_PREFIX /opt/conda/envs/flatland-rl

SHELL ["/bin/bash", "-c"]

CMD conda --version && nvidia-smi