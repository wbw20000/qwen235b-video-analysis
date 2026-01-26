# Embedding Service 镜像 (GPU)
FROM traffic-vlm-base:latest

LABEL service="embedding-service"
LABEL description="SigLIP Embedding Service with GPU support"

USER root

# PyTorch + Transformers
COPY docker/requirements-embedding.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 端口
EXPOSE 8080

# 环境变量
ENV MODEL_NAME=google/siglip-base-patch16-384
ENV MODEL_PATH=/models/siglip-base-patch16-384
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV VECTOR_TTL=3600
ENV MAX_REQUESTS=1000
ENV BATCH_SIZE=32

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python /app/healthcheck.py http://localhost:8080/health

# 启动命令
CMD ["python", "-m", "workers.embedding_service", "--host", "0.0.0.0", "--port", "8080"]
