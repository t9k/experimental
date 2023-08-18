FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

ENV NB_PREFIX=/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
  apt-get install -yq --no-install-recommends build-essential ca-certificates git software-properties-common && \
  add-apt-repository ppa:ubuntu-toolchain-r/test && \
  apt-get install -yq --no-install-recommends gcc-11 g++-11 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 90 --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11 && \
  apt-get update && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://ghproxy.com/https://github.com/oobabooga/text-generation-webui
WORKDIR /workspace/text-generation-webui
RUN pip3 install --no-cache-dir --upgrade pip && \
  pip3 install --no-cache-dir --default-timeout=100 -r requirements.txt

ENTRYPOINT [ "python", "server.py" ]
