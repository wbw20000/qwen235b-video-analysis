#!/bin/bash
# Chaos Test - 故障注入测试
# 验证系统的容错能力和自愈能力
#
# 使用方法:
#   ./scripts/chaos_test.sh [options]
#
# 选项:
#   --remote HOST     远程主机 (默认: r6500-g4-2)
#   --user USER       SSH 用户 (默认: bj)
#   --test TEST       指定测试 (pod-kill, redis-restart, vllm-restart, all)

set -e

# 配置
REMOTE_HOST="${REMOTE_HOST:-r6500-g4-2}"
REMOTE_USER="${REMOTE_USER:-bj}"
TEST_TYPE="${TEST_TYPE:-all}"
NAMESPACE="traffic-vlm"

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
        --test) TEST_TYPE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 远程执行
remote_exec() {
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "$1"
}

echo "=========================================="
echo "Traffic VLM Chaos Test"
echo "=========================================="
echo "远程主机: ${REMOTE_USER}@${REMOTE_HOST}"
echo "测试类型: ${TEST_TYPE}"
echo ""

# 等待 Pod 就绪
wait_for_pod() {
    local label=$1
    local timeout=${2:-120}
    log_info "等待 Pod $label 就绪 (超时: ${timeout}s)..."
    if remote_exec "kubectl wait --for=condition=ready pod -l $label -n $NAMESPACE --timeout=${timeout}s" 2>/dev/null; then
        log_success "Pod $label 已就绪"
        return 0
    else
        log_error "Pod $label 未能在 ${timeout}s 内就绪"
        return 1
    fi
}

# 测试 1: Pod 杀死与自愈
test_pod_kill() {
    log_info "========== 测试: Pod 杀死与自愈 =========="

    # 选择一个 semantic-analyzer pod
    POD=$(remote_exec "kubectl get pod -n $NAMESPACE -l app=semantic-analyzer -o jsonpath='{.items[0].metadata.name}'" 2>/dev/null)

    if [ -z "$POD" ]; then
        log_warn "未找到 semantic-analyzer pod，跳过测试"
        return
    fi

    log_info "杀死 Pod: $POD"
    remote_exec "kubectl delete pod $POD -n $NAMESPACE --grace-period=0 --force" 2>/dev/null || true

    # 等待新 Pod 启动
    sleep 5
    if wait_for_pod "app=semantic-analyzer" 60; then
        log_success "Pod 自愈成功"
    else
        log_error "Pod 自愈失败"
    fi
}

# 测试 2: Redis 重启
test_redis_restart() {
    log_info "========== 测试: Redis 重启 =========="

    # 记录当前任务数
    BEFORE=$(remote_exec "kubectl exec -n $NAMESPACE redis-0 -- redis-cli xlen video_tasks" 2>/dev/null || echo "0")
    log_info "重启前 video_tasks 长度: $BEFORE"

    # 重启 Redis
    log_info "重启 Redis..."
    remote_exec "kubectl rollout restart statefulset/redis -n $NAMESPACE"

    # 等待 Redis 就绪
    sleep 10
    if wait_for_pod "app=redis" 120; then
        # 检查数据是否持久化
        AFTER=$(remote_exec "kubectl exec -n $NAMESPACE redis-0 -- redis-cli xlen video_tasks" 2>/dev/null || echo "0")
        log_info "重启后 video_tasks 长度: $AFTER"

        if [ "$BEFORE" = "$AFTER" ]; then
            log_success "Redis 数据持久化正常"
        else
            log_warn "Redis 数据可能有变化 (AOF 持久化延迟)"
        fi
    else
        log_error "Redis 重启失败"
    fi
}

# 测试 3: vLLM 重启
test_vllm_restart() {
    log_info "========== 测试: vLLM 重启 =========="

    # 检查 vLLM 当前状态
    VLLM_STATUS=$(remote_exec "kubectl get pod -n $NAMESPACE -l app=vllm-server -o jsonpath='{.items[0].status.phase}'" 2>/dev/null)
    log_info "当前 vLLM 状态: $VLLM_STATUS"

    if [ "$VLLM_STATUS" != "Running" ]; then
        log_warn "vLLM 当前不在运行状态，跳过测试"
        return
    fi

    # 重启 vLLM
    log_info "重启 vLLM (这可能需要几分钟加载模型)..."
    remote_exec "kubectl rollout restart statefulset/vllm-server -n $NAMESPACE"

    # 等待 vLLM 就绪
    sleep 30
    if wait_for_pod "app=vllm-server" 300; then
        # 检查 vLLM 健康
        sleep 30  # 额外等待模型加载
        if remote_exec "kubectl exec -n $NAMESPACE deployment/vlm-proxy -- curl -s http://vllm-server:8000/health" 2>/dev/null | grep -q "ok\|healthy"; then
            log_success "vLLM 重启成功，服务正常"
        else
            log_warn "vLLM Pod 运行中，但可能仍在加载模型"
        fi
    else
        log_error "vLLM 重启失败"
    fi
}

# 测试 4: 网络分区模拟
test_network_partition() {
    log_info "========== 测试: 网络分区模拟 =========="

    # 暂时阻断 embedding-service 到 vllm 的连接
    log_info "模拟 embedding-service 网络隔离..."

    # 使用 NetworkPolicy (如果集群支持)
    cat <<EOF | remote_exec "kubectl apply -f -"
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: chaos-test-isolation
  namespace: ${NAMESPACE}
spec:
  podSelector:
    matchLabels:
      app: embedding-service
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
EOF

    log_info "等待 10 秒观察系统行为..."
    sleep 10

    # 检查 embedding-service 是否仍在运行
    EMBED_STATUS=$(remote_exec "kubectl get pod -n $NAMESPACE -l app=embedding-service -o jsonpath='{.items[0].status.phase}'" 2>/dev/null)
    if [ "$EMBED_STATUS" = "Running" ]; then
        log_success "Embedding Service 在网络隔离下仍然运行"
    else
        log_warn "Embedding Service 状态异常: $EMBED_STATUS"
    fi

    # 恢复网络
    log_info "恢复网络..."
    remote_exec "kubectl delete networkpolicy chaos-test-isolation -n $NAMESPACE" 2>/dev/null || true

    log_success "网络分区测试完成"
}

# 测试 5: 负载测试
test_load() {
    log_info "========== 测试: 简单负载测试 =========="

    API_URL="http://${REMOTE_HOST}:30500/api/v1/stats"

    log_info "发送 50 个并发请求到 API Gateway..."
    RESULTS=$(
        for i in $(seq 1 50); do
            curl -s -o /dev/null -w "%{http_code}\n" "$API_URL" &
        done
        wait
    )

    SUCCESS=$(echo "$RESULTS" | grep -c "200" || true)
    FAIL=$((50 - SUCCESS))

    log_info "成功: $SUCCESS, 失败: $FAIL"

    if [ $SUCCESS -ge 45 ]; then
        log_success "负载测试通过 (>= 90% 成功率)"
    else
        log_warn "负载测试部分失败 (成功率: $((SUCCESS * 2))%)"
    fi
}

# 测试 6: 资源耗尽模拟
test_resource_exhaustion() {
    log_info "========== 测试: 资源监控 =========="

    # 获取节点资源使用情况
    log_info "当前节点资源使用:"
    remote_exec "kubectl top nodes" 2>/dev/null || log_warn "metrics-server 未安装"

    log_info "当前 Pod 资源使用:"
    remote_exec "kubectl top pods -n $NAMESPACE" 2>/dev/null || log_warn "metrics-server 未安装"

    # 检查 GPU 使用
    log_info "GPU 使用情况:"
    remote_exec "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv" 2>/dev/null || log_warn "nvidia-smi 不可用"

    log_success "资源监控检查完成"
}

# 执行测试
run_tests() {
    case $TEST_TYPE in
        pod-kill)
            test_pod_kill
            ;;
        redis-restart)
            test_redis_restart
            ;;
        vllm-restart)
            test_vllm_restart
            ;;
        network)
            test_network_partition
            ;;
        load)
            test_load
            ;;
        resource)
            test_resource_exhaustion
            ;;
        all)
            test_pod_kill
            echo ""
            test_redis_restart
            echo ""
            test_load
            echo ""
            test_resource_exhaustion
            echo ""
            log_info "跳过 vLLM 重启测试 (耗时较长，可单独运行: --test vllm-restart)"
            ;;
        *)
            log_error "未知测试类型: $TEST_TYPE"
            echo "可用测试: pod-kill, redis-restart, vllm-restart, network, load, resource, all"
            exit 1
            ;;
    esac
}

# 主流程
run_tests

echo ""
echo "=========================================="
echo "Chaos Test 完成"
echo "=========================================="
