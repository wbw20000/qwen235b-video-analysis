# API Gateway 镜像
FROM traffic-vlm-base:latest

LABEL service="api-gateway"
LABEL description="REST API Gateway for Traffic VLM"

USER root

# Python 依赖
COPY docker/requirements-api.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 端口
EXPOSE 5000

# 环境变量
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV RESULTS_DIR=/data1/results
ENV AGGREGATED_DIR=/data1/aggregated

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py http://localhost:5000/health

# 启动命令
CMD ["python", "-m", "workers.api_gateway", "--host", "0.0.0.0", "--port", "5000"]
