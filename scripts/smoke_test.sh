#!/bin/bash
# Smoke Test - 基础功能验证
# 验证所有服务是否正常运行
#
# 使用方法:
#   ./scripts/smoke_test.sh [options]
#
# 选项:
#   --remote HOST     远程主机 (默认: r6500-g4-2)
#   --user USER       SSH 用户 (默认: bj)
#   --api-port PORT   API Gateway 端口 (默认: 30500)

set -e

# 配置
REMOTE_HOST="${REMOTE_HOST:-r6500-g4-2}"
REMOTE_USER="${REMOTE_USER:-bj}"
API_PORT="${API_PORT:-30500}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        --user) REMOTE_USER="$2"; shift 2 ;;
        --api-port) API_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 计数器
PASS=0
FAIL=0
WARN=0

check_pass() {
    PASS=$((PASS + 1))
    log_success "$1"
}

check_fail() {
    FAIL=$((FAIL + 1))
    log_error "$1"
}

check_warn() {
    WARN=$((WARN + 1))
    log_warn "$1"
}

echo "=========================================="
echo "Traffic VLM Smoke Test"
echo "=========================================="
echo "远程主机: ${REMOTE_USER}@${REMOTE_HOST}"
echo "API 端口: ${API_PORT}"
echo ""

# 远程执行
remote_exec() {
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "$1"
}

# 测试 1: SSH 连接
log_info "测试 1: SSH 连接..."
if ssh -o ConnectTimeout=5 "${REMOTE_USER}@${REMOTE_HOST}" "echo 'ok'" &>/dev/null; then
    check_pass "SSH 连接正常"
else
    check_fail "SSH 连接失败"
    exit 1
fi

# 测试 2: K8S 集群状态
log_info "测试 2: K8S 集群状态..."
if remote_exec "kubectl cluster-info" &>/dev/null; then
    check_pass "K8S 集群正常"
else
    check_fail "K8S 集群异常"
fi

# 测试 3: Namespace 存在
log_info "测试 3: Namespace..."
if remote_exec "kubectl get namespace traffic-vlm" &>/dev/null; then
    check_pass "Namespace traffic-vlm 存在"
else
    check_fail "Namespace traffic-vlm 不存在"
fi

# 测试 4: Pod 状态
log_info "测试 4: Pod 状态..."
echo ""
PODS=$(remote_exec "kubectl get pods -n traffic-vlm -o jsonpath='{range .items[*]}{.metadata.name} {.status.phase}{\"\\n\"}{end}'")
echo "$PODS" | while read name phase; do
    if [ -n "$name" ]; then
        if [ "$phase" = "Running" ]; then
            check_pass "Pod $name: $phase"
        elif [ "$phase" = "Pending" ]; then
            check_warn "Pod $name: $phase (等待中)"
        else
            check_fail "Pod $name: $phase"
        fi
    fi
done

# 测试 5: Redis 连接
log_info "测试 5: Redis 连接..."
if remote_exec "kubectl exec -n traffic-vlm redis-0 -- redis-cli ping" 2>/dev/null | grep -q "PONG"; then
    check_pass "Redis 响应正常"
else
    check_fail "Redis 无响应"
fi

# 测试 6: API Gateway 健康检查
log_info "测试 6: API Gateway 健康检查..."
if curl -s -o /dev/null -w "%{http_code}" "http://${REMOTE_HOST}:${API_PORT}/health" | grep -q "200"; then
    check_pass "API Gateway /health 返回 200"
else
    check_warn "API Gateway 未就绪或无法访问"
fi

# 测试 7: vLLM 服务状态
log_info "测试 7: vLLM 服务状态..."
VLLM_STATUS=$(remote_exec "kubectl get pod -n traffic-vlm -l app=vllm-server -o jsonpath='{.items[0].status.phase}'" 2>/dev/null)
if [ "$VLLM_STATUS" = "Running" ]; then
    # 检查 vLLM 是否真正就绪
    if remote_exec "kubectl exec -n traffic-vlm deployment/vlm-proxy -- curl -s http://vllm-server:8000/health" 2>/dev/null | grep -q "ok\|healthy"; then
        check_pass "vLLM 服务就绪"
    else
        check_warn "vLLM Pod 运行中，但服务可能仍在加载模型"
    fi
else
    check_warn "vLLM Pod 状态: $VLLM_STATUS"
fi

# 测试 8: Embedding Service
log_info "测试 8: Embedding Service..."
EMBED_STATUS=$(remote_exec "kubectl get pod -n traffic-vlm -l app=embedding-service -o jsonpath='{.items[0].status.phase}'" 2>/dev/null)
if [ "$EMBED_STATUS" = "Running" ]; then
    check_pass "Embedding Service 运行中"
else
    check_warn "Embedding Service 状态: $EMBED_STATUS"
fi

# 测试 9: GPU 分配
log_info "测试 9: GPU 分配..."
GPU_COUNT=$(remote_exec "kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\\.com/gpu}'" 2>/dev/null)
if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 0 ]; then
    check_pass "可用 GPU 数量: $GPU_COUNT"
else
    check_warn "无法获取 GPU 信息或无可用 GPU"
fi

# 测试 10: 存储挂载
log_info "测试 10: 存储挂载..."
if remote_exec "[ -d /data/models ] && [ -d /data1/videos ]" 2>/dev/null; then
    check_pass "存储目录存在"
else
    check_fail "存储目录缺失"
fi

# 测试 11: 模型文件
log_info "测试 11: 模型文件..."
if remote_exec "[ -d /data/models/qwen3-vl-32b-awq ]" 2>/dev/null; then
    check_pass "Qwen3-VL 模型存在"
else
    check_warn "Qwen3-VL 模型未下载"
fi

if remote_exec "[ -d /data/models/siglip-base-patch16-384 ]" 2>/dev/null; then
    check_pass "SigLIP 模型存在"
else
    check_warn "SigLIP 模型未下载"
fi

# 测试 12: API 功能测试
log_info "测试 12: API 功能测试..."
STATS=$(curl -s "http://${REMOTE_HOST}:${API_PORT}/api/v1/stats" 2>/dev/null)
if echo "$STATS" | grep -q "queues"; then
    check_pass "API /api/v1/stats 正常"
else
    check_warn "API 统计接口异常"
fi

echo ""
echo "=========================================="
echo "Smoke Test 结果"
echo "=========================================="
echo -e "通过: ${GREEN}${PASS}${NC}"
echo -e "警告: ${YELLOW}${WARN}${NC}"
echo -e "失败: ${RED}${FAIL}${NC}"
echo ""

if [ $FAIL -gt 0 ]; then
    log_error "存在失败项，请检查日志"
    exit 1
elif [ $WARN -gt 0 ]; then
    log_warn "存在警告项，部分服务可能需要等待就绪"
    exit 0
else
    log_success "所有测试通过!"
    exit 0
fi
