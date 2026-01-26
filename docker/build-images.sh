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

# 1. 构建基础镜像
echo ""
echo "[1/8] 构建基础镜像..."
docker build \
    -t traffic-vlm-base:latest \
    -f docker/base.Dockerfile \
    .

# 2. 构建 API Gateway
echo ""
echo "[2/8] 构建 API Gateway..."
docker build \
    -t ${REGISTRY}/api-gateway:latest \
    -f docker/api-gateway.Dockerfile \
    .

# 3. 构建 Embedding Service
echo ""
echo "[3/8] 构建 Embedding Service..."
docker build \
    -t ${REGISTRY}/embedding-service:latest \
    -f docker/embedding-service.Dockerfile \
    .

# 4. 构建 VLM Proxy
echo ""
echo "[4/8] 构建 VLM Proxy..."
docker build \
    -t ${REGISTRY}/vlm-proxy:latest \
    -f docker/vlm-proxy.Dockerfile \
    .

# 5. 构建 Semantic Analyzer
echo ""
echo "[5/8] 构建 Semantic Analyzer..."
docker build \
    -t ${REGISTRY}/semantic-analyzer:latest \
    -f docker/semantic-analyzer.Dockerfile \
    .

# 6. 构建 RTSP Ingest
echo ""
echo "[6/8] 构建 RTSP Ingest..."
docker build \
    -t ${REGISTRY}/rtsp-ingest:latest \
    -f docker/rtsp-ingest.Dockerfile \
    .

# 7. 构建 Dispatcher
echo ""
echo "[7/8] 构建 Dispatcher..."
docker build \
    -t ${REGISTRY}/dispatcher:latest \
    -f docker/dispatcher.Dockerfile \
    .

# 8. 构建 Aggregator
echo ""
echo "[8/8] 构建 Aggregator..."
docker build \
    -t ${REGISTRY}/aggregator:latest \
    -f docker/aggregator.Dockerfile \
    .

echo ""
echo "=========================================="
echo "所有镜像构建完成!"
echo "=========================================="
echo ""
echo "镜像列表:"
docker images | grep -E "(traffic-vlm|api-gateway|embedding|vlm-proxy|semantic|rtsp-ingest|dispatcher|aggregator)" | head -20

echo ""
echo "如需推送到 registry，请运行:"
echo "  docker push ${REGISTRY}/api-gateway:latest"
echo "  docker push ${REGISTRY}/embedding-service:latest"
echo "  docker push ${REGISTRY}/vlm-proxy:latest"
echo "  docker push ${REGISTRY}/semantic-analyzer:latest"
echo "  docker push ${REGISTRY}/rtsp-ingest:latest"
echo "  docker push ${REGISTRY}/dispatcher:latest"
echo "  docker push ${REGISTRY}/aggregator:latest"
