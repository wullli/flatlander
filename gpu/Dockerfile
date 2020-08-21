# FROM continuumio/miniconda:latest
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update
RUN apt-get install -y wget python-opengl && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"

COPY ./environment-gpu.yml /src/environment-gpu.yml
RUN ls /src
RUN conda env create -f /src/environment-gpu.yml

ENV PATH /root/miniconda3/envs/flatland-rl/bin:$PATH
ENV CONDA_DEFAULT_ENV flatland-rl
ENV CONDA_PREFIX /root/miniconda3/envs/flatland-rl

SHELL ["/bin/bash", "-c"]

CMD conda --version && nvidia-smi