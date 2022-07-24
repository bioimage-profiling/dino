FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

# Install python
RUN apt update \
    && apt install -y --no-install-recommends python3 python3-pip \
    && ln -sf python3 /usr/bin/python \
    && ln -sf pip3 /usr/bin/pip \
    && pip install --upgrade pip \
    && pip install wheel setuptools

RUN apt install -y --no-install-recommends git

# Install required Python libraries
RUN python -m pip install torch torchvision torchsummary --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116

RUN python -m pip install timm

# Clone repository
RUN mkdir /src && \
    cd /src && \
    git clone https://github.com/bioimage-profiling/dino.git

ENTRYPOINT ["python", "/src/dino/main_dino.py"]
