FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116

WORKDIR /src

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY . .

RUN pip3 install -e . && pip3 install -e .[dev]

