# 电动自行车违法检测分析器镜像
FROM traffic-vlm-base:latest

LABEL service="ebike-violation-analyzer"
LABEL description="E-bike Violation Analyzer"
LABEL analysis-type="ebike_violation"

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
ENV CLIP_SCORE_THRESHOLD=0.35
ENV FFMPEG_NVDEC_ENABLED=true
ENV FFMPEG_NVDEC_DEVICE=5

# 启动命令
CMD ["python", "-m", "workers.ebike_violation_analyzer"]
