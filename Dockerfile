FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl wget \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev \
        libglfw3-dev libglew-dev libosmesa6-dev \
        libx11-6 libxcursor1 libxrandr2 libxinerama1 libxi6 \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128

RUN mkdir -p /opt/src && \
    git clone https://github.com/ARISE-Initiative/robomimic.git /opt/src/robomimic && \
    cd /opt/src/robomimic && git checkout e10526b9a40c78b41f1e37e60041dc0ec0a5f60f && \
    git clone https://github.com/ARISE-Initiative/robosuite.git /opt/src/robosuite && \
    cd /opt/src/robosuite && git checkout 6c10ef24a4bb52f59199976125060ce793470e6e

COPY patches/ /tmp/patches/
RUN cd /opt/src/robomimic && git apply /tmp/patches/robomimic.patch && \
    cd /opt/src/robosuite && git apply /tmp/patches/robosuite.patch && \
    rm -rf /tmp/patches

RUN pip install --no-cache-dir -e /opt/src/robomimic -e /opt/src/robosuite

COPY . /workspace

CMD ["bash"]
