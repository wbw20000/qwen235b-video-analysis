# RTSP Ingest 镜像 (CPU)
FROM traffic-vlm-base:latest

LABEL service="rtsp-ingest"
LABEL description="RTSP stream recording with 10-min segments"

USER root

# Python 依赖
COPY docker/requirements-ingest.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV CAMERA_ID=unknown
ENV RTSP_URL=
ENV OUTPUT_DIR=/data1/videos/rtsp_recordings
ENV SEGMENT_DURATION=600
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# 无端口暴露（只写文件）

# 启动命令（需通过环境变量或参数指定 camera-id 和 rtsp-url）
CMD ["python", "-m", "workers.rtsp_ingest"]
