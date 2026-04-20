# RTSP Ingest 专用轻量镜像
# 仅需 ffmpeg + Python, 无需 CUDA/PyTorch
FROM python:3.10-slim

LABEL service="rtsp-ingest"
LABEL description="RTSP stream recording with ffmpeg remux"
LABEL version="v2"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Shanghai

# 系统依赖: 只装 ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# 复制代码 (rtsp_ingest 不需要 pip 依赖, 全是标准库)
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV CAMERA_ID=unknown
ENV RTSP_URL=
ENV OUTPUT_DIR=/data1/videos/rtsp_recordings
ENV SEGMENT_DURATION=60

CMD ["python", "-m", "workers.rtsp_ingest"]
