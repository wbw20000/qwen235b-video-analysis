#!/bin/bash
# Traffic VLM 远程部署脚本
# 一键部署 K8S 微服务架构到 T4 服务器
#
# 使用方法:
#   ./scripts/deploy_remote.sh [options]
#
# 选项:
#   --remote HOST     远程主机 (默认: r6500-g4-2)
#   --user USER       SSH 用户 (默认: bj)
#   --skip-k8s        跳过 K8S 安装
#   --skip-models     跳过模型下载
#   --skip-build      跳过镜像构建
#   --dry-run         只打印命令，不执行

set -e

# 配置
REMOTE_HOST="${REMOTE_HOST:-r6500-g4-2}"
REMOTE_USER="${REMOTE_USER:-bj}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SKIP_K8S=false
SKIP_MODELS=false
SKIP_BUILD=false
DRY_RUN=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        --user) REMOTE_USER="$2"; shift 2 ;;
        --skip-k8s) SKIP_K8S=true; shift ;;
        --skip-models) SKIP_MODELS=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 执行远程命令
remote_exec() {
    local cmd="$1"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] ssh ${REMOTE_USER}@${REMOTE_HOST} '$cmd'"
    else
        ssh "${REMOTE_USER}@${REMOTE_HOST}" "$cmd"
    fi
}

# 复制文件到远程
remote_copy() {
    local src="$1"
    local dst="$2"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] scp -r $src ${REMOTE_USER}@${REMOTE_HOST}:$dst"
    else
        scp -r "$src" "${REMOTE_USER}@${REMOTE_HOST}:$dst"
    fi
}

echo "=========================================="
echo "Traffic VLM 远程部署"
echo "=========================================="
echo "远程主机: ${REMOTE_USER}@${REMOTE_HOST}"
echo "项目目录: ${PROJECT_ROOT}"
echo ""

# 步骤 1: 检查远程连接
log_info "步骤 1/8: 检查远程连接..."
if ! ssh -o ConnectTimeout=5 "${REMOTE_USER}@${REMOTE_HOST}" "echo 'SSH OK'" 2>/dev/null; then
    log_error "无法连接到远程服务器"
    exit 1
fi
log_success "远程连接正常"

# 步骤 2: 创建目录结构
log_info "步骤 2/8: 创建目录结构..."
remote_exec "sudo mkdir -p /data/{models,app,cache,logs} /data1/{videos,results,aggregated}"
remote_exec "sudo chown -R ${REMOTE_USER}:${REMOTE_USER} /data /data1"
log_success "目录结构创建完成"

# 步骤 3: 安装 Docker
log_info "步骤 3/8: 检查/安装 Docker..."
if ! remote_exec "docker --version" 2>/dev/null; then
    log_warn "Docker 未安装，开始安装..."
    remote_exec "curl -fsSL https://get.docker.com | sudo sh"
    remote_exec "sudo usermod -aG docker ${REMOTE_USER}"
    log_success "Docker 安装完成"
else
    log_success "Docker 已安装"
fi

# 步骤 4: 安装 K8S (可选)
if [ "$SKIP_K8S" = false ]; then
    log_info "步骤 4/8: 检查/安装 K8S..."
    if ! remote_exec "kubectl version --client" 2>/dev/null; then
        log_warn "K8S 未安装，开始安装..."

        # 安装 containerd
        remote_exec "sudo apt-get update && sudo apt-get install -y containerd"
        remote_exec "sudo mkdir -p /etc/containerd && containerd config default | sudo tee /etc/containerd/config.toml"
        remote_exec "sudo systemctl restart containerd"

        # 安装 kubeadm, kubelet, kubectl
        remote_exec "sudo apt-get install -y apt-transport-https ca-certificates curl gpg"
        remote_exec "curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.29/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg"
        remote_exec "echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.29/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list"
        remote_exec "sudo apt-get update && sudo apt-get install -y kubelet kubeadm kubectl"
        remote_exec "sudo apt-mark hold kubelet kubeadm kubectl"

        # 初始化集群
        remote_exec "sudo kubeadm init --pod-network-cidr=10.244.0.0/16"
        remote_exec "mkdir -p \$HOME/.kube && sudo cp -i /etc/kubernetes/admin.conf \$HOME/.kube/config && sudo chown \$(id -u):\$(id -g) \$HOME/.kube/config"

        # 允许 master 调度
        remote_exec "kubectl taint nodes --all node-role.kubernetes.io/control-plane-"

        # 安装网络插件
        remote_exec "kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml"

        log_success "K8S 安装完成"
    else
        log_success "K8S 已安装"
    fi

    # 安装 NVIDIA Device Plugin
    log_info "安装 NVIDIA Device Plugin..."
    if ! remote_exec "kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset" 2>/dev/null; then
        remote_exec "kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml"
        log_success "NVIDIA Device Plugin 安装完成"
    else
        log_success "NVIDIA Device Plugin 已安装"
    fi
else
    log_info "步骤 4/8: 跳过 K8S 安装"
fi

# 步骤 5: 下载模型 (可选)
if [ "$SKIP_MODELS" = false ]; then
    log_info "步骤 5/8: 下载模型..."

    # 检查模型是否存在
    if ! remote_exec "[ -d /data/models/qwen3-vl-32b-awq ]"; then
        log_warn "下载 Qwen3-VL-32B-AWQ 模型..."
        remote_exec "pip install huggingface_hub"
        remote_exec "huggingface-cli download Qwen/Qwen3-VL-32B-Instruct-AWQ --local-dir /data/models/qwen3-vl-32b-awq"
        log_success "Qwen3-VL 模型下载完成"
    else
        log_success "Qwen3-VL 模型已存在"
    fi

    if ! remote_exec "[ -d /data/models/siglip-base-patch16-384 ]"; then
        log_warn "下载 SigLIP 模型..."
        remote_exec "huggingface-cli download google/siglip-base-patch16-384 --local-dir /data/models/siglip-base-patch16-384"
        log_success "SigLIP 模型下载完成"
    else
        log_success "SigLIP 模型已存在"
    fi
else
    log_info "步骤 5/8: 跳过模型下载"
fi

# 步骤 6: 同步代码
log_info "步骤 6/8: 同步代码到远程..."
remote_copy "${PROJECT_ROOT}/workers" "/data/app/"
remote_copy "${PROJECT_ROOT}/docker" "/data/app/"
remote_copy "${PROJECT_ROOT}/k8s" "/data/app/"
remote_copy "${PROJECT_ROOT}/scripts" "/data/app/"
log_success "代码同步完成"

# 步骤 7: 构建镜像 (可选)
if [ "$SKIP_BUILD" = false ]; then
    log_info "步骤 7/8: 构建 Docker 镜像..."
    remote_exec "cd /data/app && chmod +x docker/build-images.sh && ./docker/build-images.sh localhost:5000"
    log_success "镜像构建完成"
else
    log_info "步骤 7/8: 跳过镜像构建"
fi

# 步骤 8: 部署 K8S 服务
log_info "步骤 8/8: 部署 K8S 服务..."

# 创建 namespace
remote_exec "kubectl apply -f /data/app/k8s/00-namespace.yaml"

# 部署 Redis
remote_exec "kubectl apply -f /data/app/k8s/01-redis.yaml"
log_info "等待 Redis 就绪..."
remote_exec "kubectl wait --for=condition=ready pod -l app=redis -n traffic-vlm --timeout=120s" || true

# 部署 vLLM
remote_exec "kubectl apply -f /data/app/k8s/02-vllm-server.yaml"
log_info "等待 vLLM 就绪 (可能需要几分钟)..."

# 部署其他服务
remote_exec "kubectl apply -f /data/app/k8s/03-embedding-service.yaml"
remote_exec "kubectl apply -f /data/app/k8s/04-vlm-proxy.yaml"
remote_exec "kubectl apply -f /data/app/k8s/05-semantic-analyzer.yaml"
remote_exec "kubectl apply -f /data/app/k8s/06-aggregator.yaml"
remote_exec "kubectl apply -f /data/app/k8s/07-api-gateway.yaml"

# 部署 RTSP Ingest (可选，需要网络连通)
# remote_exec "kubectl apply -f /data/app/k8s/08-rtsp-ingest.yaml"

log_success "K8S 服务部署完成"

echo ""
echo "=========================================="
echo "部署完成!"
echo "=========================================="
echo ""
echo "查看 Pod 状态:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'kubectl get pods -n traffic-vlm'"
echo ""
echo "查看服务状态:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST} 'kubectl get svc -n traffic-vlm'"
echo ""
echo "API Gateway 地址:"
echo "  http://${REMOTE_HOST}:30500"
echo ""
echo "运行 smoke test:"
echo "  ./scripts/smoke_test.sh --remote ${REMOTE_HOST}"
