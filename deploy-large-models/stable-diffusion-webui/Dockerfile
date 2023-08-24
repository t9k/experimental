FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

ENV NB_PREFIX=/

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -yq --no-install-recommends \
  ffmpeg \
  libsm6 \
  libxext6 \
  git \
  && apt-get clean && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY sd-webui-aki-v4 ./sd-webui-aki-v4
WORKDIR /workspace/sd-webui-aki-v4
RUN mv .cache /root

RUN pip install --no-cache-dir --upgrade pip && \
  pip install --no-cache-dir --default-timeout=100 -r requirements_versions.txt
RUN python launch.py --exit --skip-torch-cuda-test

ENTRYPOINT [ "python", "launch.py" ]
