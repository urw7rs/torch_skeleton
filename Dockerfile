FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    python3-setuptools \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 

RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH /root/miniconda3/bin:$PATH

ENV CONDA_PREFIX /root/miniconda3/envs/torch_skeleton

# Clear .bashrc (it refuses to run non-interactively otherwise).
RUN echo > ~/.bashrc

# Add conda logic to .bashrc.
RUN conda init bash

# Create new environment and install some dependencies.
RUN conda create -y -n torch_skeleton python=3.8 

# Activate environment in .bashrc.
RUN echo "conda activate torch_skeleton" >> /root/.bashrc

# Make bash excecute .bashrc even when running non-interactively.
ENV BASH_ENV /root/.bashrc

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

COPY .git .git

RUN git restore .

RUN pip install -e . && pip install -e .[examples]

CMD ["bash"]
