FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV NB_PREFIX=/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -yq --no-install-recommends \
 ffmpeg \
 libsm6 \
 libxext6 \
 build-essential \
 python3-opencv \
 libopencv-dev \
 && apt-get clean && \
 rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --default-timeout=100 \
  "InvokeAI[xformers]" --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu117 \
  pypatchmatch

COPY run.sh .
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5
ENTRYPOINT [ "./run.sh" ]
