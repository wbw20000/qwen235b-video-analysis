# PreProcessor 镜像 (CUDA MOG2 + SigLIP)
# 共享预处理服务: FFmpeg NVDEC + CUDA MOG2 + SigLIP Embedding
FROM opencv-cuda-base:latest

LABEL service="preprocessor"
LABEL description="Shared Preprocessing Service - MOG2 + SigLIP"

USER root

# PyTorch (CUDA 12.1) + 其他依赖
COPY docker/requirements-preprocessor.txt /tmp/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV RESULTS_DIR=/data1/results
ENV MAX_TASKS=100
ENV CACHE_TTL=300

# SigLIP 模型路径 (从 PVC 挂载)
ENV SIGLIP_MODEL_PATH=/data/models/siglip-base-patch16-384

# CUDA 设备 (由 K8S 分配)
ENV CUDA_DEVICE=0

# 健康检查 (检查 Redis 连接)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import redis; r=redis.Redis(host='${REDIS_HOST}', port=${REDIS_PORT}); r.ping()"

# 启动命令
CMD ["python", "-m", "workers.preprocessor"]
