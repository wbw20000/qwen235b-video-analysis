# Dispatcher 镜像 (CPU - 阶段1.5 核心)
FROM traffic-vlm-base:latest

LABEL service="dispatcher"
LABEL description="Dispatcher - overlapping window generation (Phase 1.5)"

USER root

# Python 依赖
COPY docker/requirements-dispatcher.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV CAMERA_ID=unknown
ENV INPUT_DIR=/data1/videos/rtsp_recordings
ENV OUTPUT_DIR=/data1/videos/windows
ENV OVERLAP_SEC=30
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# 无端口暴露（只处理文件 + 写 Redis）

# 启动命令
CMD ["python", "-m", "workers.dispatcher"]
