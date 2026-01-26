# Aggregator 镜像 (CPU - 结果去重)
FROM traffic-vlm-base:latest

LABEL service="aggregator"
LABEL description="Aggregator - result deduplication and aggregation"

USER root

# Python 依赖
COPY docker/requirements-aggregator.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV RESULTS_DIR=/data1/results
ENV AGGREGATED_DIR=/data1/aggregated
ENV DEDUP_WINDOW_SEC=15
ENV MAX_TASKS=100

# 无端口暴露（纯消费者）

# 启动命令
CMD ["python", "-m", "workers.aggregator"]
