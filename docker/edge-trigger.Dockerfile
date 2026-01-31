# Edge Trigger 镜像 - 边缘端事故预检测
# 注意: 边缘部署可能使用 CPU-only 版本
FROM traffic-vlm-base:latest

LABEL service="edge-trigger"
LABEL description="Edge Trigger - lightweight accident pre-detection at edge"
LABEL tier="edge"

USER root

# Python 依赖
COPY docker/requirements-edge-trigger.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 复制代码
COPY --chown=appuser:appuser workers/ /app/workers/

USER appuser

# 环境变量
ENV CAMERA_ID=unknown
ENV CENTER_API_URL=http://api-gateway:5000
ENV LOCAL_EMBEDDING_URL=""

# 边缘触发阈值 (低阈值保证高召回)
ENV EDGE_SIMILARITY_THRESHOLD=0.12
ENV EDGE_WINDOW_SIZE_SEC=60
ENV EDGE_SLIDE_INTERVAL_SEC=5
ENV EDGE_FRAME_SAMPLE_FPS=1.0
ENV EDGE_MAX_KEYFRAMES=5

# ONNX 模式 (CPU-only)
ENV EDGE_USE_ONNX=false
ENV EDGE_ONNX_MODEL_PATH=/models/siglip_int8.onnx

# 自愈配置
ENV MAX_TRIGGERS=1000
ENV POLL_INTERVAL_SEC=5

# 无端口暴露（轮询窗口 + POST 到中心云）

# 启动命令
CMD ["python", "-m", "workers.edge_trigger"]
