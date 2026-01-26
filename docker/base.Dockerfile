# 基础镜像 - 所有服务的公共依赖
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Traffic VLM Team"
LABEL description="Base image for Traffic Accident Analysis System"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Shanghai

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# 升级 pip
RUN python -m pip install --upgrade pip setuptools wheel

# 创建工作目录
WORKDIR /app

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# 默认用户
USER appuser

# 健康检查脚本
COPY --chown=appuser:appuser docker/healthcheck.py /app/healthcheck.py

CMD ["python", "--version"]
