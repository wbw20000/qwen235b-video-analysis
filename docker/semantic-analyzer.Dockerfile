# Semantic Analyzer 镜像 (CPU - 唯一编排者)
FROM traffic-vlm-base:latest

LABEL service="semantic-analyzer"
LABEL description="Semantic Analyzer - the sole orchestrator"

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

# 无端口暴露（纯消费者）

# 启动命令
CMD ["python", "-m", "workers.semantic_analyzer"]
