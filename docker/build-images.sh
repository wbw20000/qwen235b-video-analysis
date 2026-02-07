#!/bin/bash
# 构建所有 Docker 镜像
# 使用方法: ./build-images.sh [registry]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REGISTRY="${1:-localhost:5000}"

echo "=========================================="
echo "Traffic VLM Docker 镜像构建"
echo "Registry: $REGISTRY"
echo "=========================================="

cd "$PROJECT_ROOT"

# 0. 构建 OpenCV CUDA 基础镜像 (约15-20分钟，首次构建后有缓存)
echo ""
echo "[0/9] 构建 OpenCV CUDA 基础镜像..."
docker build \
    -t opencv-cuda-base:latest \
    -f docker/opencv-cuda.Dockerfile \
    .

# 1. 构建通用基础镜像
echo ""
echo "[1/9] 构建通用基础镜像..."
docker build \
    -t traffic-vlm-base:latest \
    -f docker/base.Dockerfile \
    .

# 2. 构建 API Gateway
echo ""
echo "[2/9] 构建 API Gateway..."
docker build \
    -t ${REGISTRY}/api-gateway:latest \
    -f docker/api-gateway.Dockerfile \
    .

# 3. 构建 Embedding Service
echo ""
echo "[3/9] 构建 Embedding Service..."
docker build \
    -t ${REGISTRY}/embedding-service:latest \
    -f docker/embedding-service.Dockerfile \
    .

# 4. 构建 VLM Proxy
echo ""
echo "[4/9] 构建 VLM Proxy..."
docker build \
    -t ${REGISTRY}/vlm-proxy:latest \
    -f docker/vlm-proxy.Dockerfile \
    .

# 5. 构建 Semantic Analyzer (基于 opencv-cuda-base)
echo ""
echo "[5/9] 构建 Semantic Analyzer..."
docker build \
    -t ${REGISTRY}/semantic-analyzer:latest \
    -f docker/semantic-analyzer.Dockerfile \
    .

# 6. 构建 RTSP Ingest
echo ""
echo "[6/9] 构建 RTSP Ingest..."
docker build \
    -t ${REGISTRY}/rtsp-ingest:latest \
    -f docker/rtsp-ingest.Dockerfile \
    .

# 7. 构建 Dispatcher
echo ""
echo "[7/9] 构建 Dispatcher..."
docker build \
    -t ${REGISTRY}/dispatcher:latest \
    -f docker/dispatcher.Dockerfile \
    .

# 8. 构建 Aggregator
echo ""
echo "[8/9] 构建 Aggregator..."
docker build \
    -t ${REGISTRY}/aggregator:latest \
    -f docker/aggregator.Dockerfile \
    .

# 9. 标记分析器镜像 (共用 semantic-analyzer 镜像，K8S 通过 command 区分)
echo ""
echo "[9/9] 标记分析器镜像..."
docker tag ${REGISTRY}/semantic-analyzer:latest ${REGISTRY}/mv-violation-analyzer:latest
docker tag ${REGISTRY}/semantic-analyzer:latest ${REGISTRY}/ebike-violation-analyzer:latest
docker tag ${REGISTRY}/semantic-analyzer:latest ${REGISTRY}/ads-behavior-analyzer:latest

echo ""
echo "=========================================="
echo "所有镜像构建完成!"
echo "=========================================="
echo ""
echo "镜像列表:"
docker images | grep -E "(opencv-cuda|traffic-vlm|api-gateway|embedding|vlm-proxy|semantic|rtsp-ingest|dispatcher|aggregator)" | head -20

echo ""
echo "如需推送到 registry，请运行:"
echo "  docker push ${REGISTRY}/api-gateway:latest"
echo "  docker push ${REGISTRY}/embedding-service:latest"
echo "  docker push ${REGISTRY}/vlm-proxy:latest"
echo "  docker push ${REGISTRY}/semantic-analyzer:latest"
echo "  docker push ${REGISTRY}/rtsp-ingest:latest"
echo "  docker push ${REGISTRY}/dispatcher:latest"
echo "  docker push ${REGISTRY}/aggregator:latest"
