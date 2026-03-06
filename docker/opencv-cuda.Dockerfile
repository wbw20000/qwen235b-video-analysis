# ============ Stage 1: 编译 OpenCV CUDA ============
# 使用 devel 镜像（包含 nvcc 和 CUDA 头文件）
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS opencv-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential pkg-config git \
    python3.10-dev python3-numpy python3-pip \
    libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone OpenCV 4.10.0 + contrib (depth=1 加速)
RUN git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv.git /opencv && \
    git clone --branch 4.10.0 --depth 1 https://github.com/opencv/opencv_contrib.git /opencv_contrib

# 编译 OpenCV CUDA
# - CUDA_ARCH_BIN=7.5 → Tesla T4
# - WITH_NVCUVID=OFF → nvcuvid.h 不可用 (需 Video Codec SDK)
# - BUILD_opencv_cudacodec=OFF → 同上
# - BUILD_LIST 仅编译需要的模块，减少编译时间
RUN cd /opencv && mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    -DWITH_CUDA=ON \
    -DCUDA_ARCH_BIN="7.5" \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DWITH_NVCUVID=OFF \
    -DBUILD_opencv_cudacodec=OFF \
    -DBUILD_opencv_python3=ON \
    -DPYTHON3_EXECUTABLE=/usr/bin/python3.10 \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_LIST=core,imgproc,imgcodecs,videoio,cudabgsegm,cudafilters,cudaimgproc,cudev,python3 \
    && make -j$(nproc) && make install

# ============ Stage 2: 最终运行时镜像 ============
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Shanghai

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-numpy \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    libjpeg8 libpng16-16 libtiff5 libtbb2 \
    libavcodec58 libavformat58 libswscale5 \
    curl wget git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 从构建阶段复制编译好的 OpenCV
COPY --from=opencv-builder /usr/local/lib/python3.10/dist-packages/cv2 \
     /usr/local/lib/python3.10/dist-packages/cv2
COPY --from=opencv-builder /usr/local/lib/libopencv* /usr/local/lib/
RUN ldconfig

# 创建工作目录和用户
WORKDIR /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# 健康检查脚本
COPY --chown=appuser:appuser docker/healthcheck.py /app/healthcheck.py

USER appuser

CMD ["python", "--version"]
