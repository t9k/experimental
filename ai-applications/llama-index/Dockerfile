FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 \
    llama-index==0.8.14 \
    sentence_transformers \
    accelerate \
    bitsandbytes \
    protobuf \
    qdrant_client \
    pypdf \
    lark \
    starlette \
    uvicorn

WORKDIR /workspace

COPY docs ./docs
COPY server.py ./server.py
ENTRYPOINT [ "uvicorn", "server:app" ]
