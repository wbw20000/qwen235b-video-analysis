# 自动驾驶行为检测分析器镜像 (CUDA MOG2 + FFmpeg NVDEC)
FROM opencv-cuda-base:latest

LABEL service="ads-behavior-analyzer"
LABEL description="Autonomous Driving System Behavior Analyzer"
LABEL analysis-type="ads_behavior"

USER root

# Python 依赖
COPY docker/requirements-analyzer.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV EMBEDDING_SERVICE_URL=http://embedding-service:8080
ENV VLM_PROXY_URL=http://vlm-proxy:8001
ENV RESULTS_DIR=/data1/results
ENV MAX_TASKS=50
ENV CLIP_SCORE_THRESHOLD=0.30

# 启动命令
CMD ["python", "-m", "workers.ads_behavior_analyzer"]
