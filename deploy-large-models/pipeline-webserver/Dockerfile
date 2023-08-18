FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip3 install --no-cache-dir --upgrade pip && \
  pip3 install --no-cache-dir --default-timeout=100 \
    protobuf==3.20.3 \
    sentencepiece \
    starlette \
    tokenizers \
    transformers \
    uvicorn

WORKDIR /workspace
COPY server.py .
ENTRYPOINT [ "uvicorn", "server:app" ]
