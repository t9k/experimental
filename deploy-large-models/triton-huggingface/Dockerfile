FROM nvcr.io/nvidia/tritonserver:23.05-py3

RUN pip3 install --no-cache-dir -i https://pypi.douban.com/simple --upgrade pip && \
  pip3 install --no-cache-dir -i https://pypi.douban.com/simple --default-timeout=100 \
    torch \
    torchvision \
    transformers \
    Image

ENTRYPOINT [ "tritonserver", "--model-repository=/models" ]
