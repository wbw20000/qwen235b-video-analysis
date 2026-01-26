# VLM Proxy 镜像 (并发阀门)
FROM traffic-vlm-base:latest

LABEL service="vlm-proxy"
LABEL description="VLM Proxy with concurrency control"

USER root

# Python 依赖
COPY docker/requirements-vlm-proxy.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 端口
EXPOSE 8001

# 环境变量
ENV VLLM_BASE_URL=http://vllm-server:8000/v1
ENV MAX_CONCURRENT=2
ENV REQUEST_TIMEOUT=120
ENV MAX_RETRIES=2
ENV RETRY_DELAY=2.0

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py http://localhost:8001/health

# 启动命令
CMD ["python", "-m", "workers.vlm_proxy", "--host", "0.0.0.0", "--port", "8001"]
