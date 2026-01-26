# Traffic VLM 远程部署报告

> **版本**: v1.0.0
> **分支**: 远程服务器部署V1
> **日期**: 2025-01-26
> **目标服务器**: r6500-g4-2 (8×T4 GPU)

---

## 执行摘要

### 部署目标
将交通事故视频分析系统部署为 K8S 微服务架构，实现：
- P0 稳定性要求（编排统一、文件幂等、可观测、可恢复）
- 阶段1.5 重叠窗口策略（10分钟分片 + 30秒重叠 + 去重）
- 50+ 路摄像头扩展能力

### 完成状态

| 交付物 | 状态 | 路径 |
|--------|------|------|
| 微服务代码 | ✅ 完成 | `workers/` |
| Docker 镜像定义 | ✅ 完成 | `docker/` |
| K8S 部署清单 | ✅ 完成 | `k8s/` |
| 一键部署脚本 | ✅ 完成 | `scripts/deploy_remote.sh` |
| Smoke Test 脚本 | ✅ 完成 | `scripts/smoke_test.sh` |
| Chaos Test 脚本 | ✅ 完成 | `scripts/chaos_test.sh` |
| 环境配置 | ✅ 完成 | `deploy/remote_env.yaml` |

---

## 架构设计

### 微服务拆分

```
┌─────────────────────────────────────────────────────────────────┐
│                    K8S Cluster (traffic-vlm namespace)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  API Gateway ──► Redis Streams ──► Semantic Analyzer            │
│    (HTTP)         (video_tasks)       (唯一编排者)               │
│                   (result_tasks)           │                     │
│                                            ├──► Embedding Svc    │
│                                            │      (GPU: SigLIP)  │
│                                            │                     │
│                                            └──► VLM Proxy        │
│                                                   │              │
│                                                   ▼              │
│  RTSP Ingest ──► Dispatcher ──────────────► vLLM Server         │
│   (录制分片)     (重叠窗口)                   (GPU: 2×T4)        │
│                                                   │              │
│                                                   ▼              │
│                                            Aggregator            │
│                                            (结果去重)             │
└─────────────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 职责 | 资源 | 扩展策略 |
|------|------|------|----------|
| **api-gateway** | REST API, trace_id 生成 | 1 CPU, 1GB | HPA 2-10 |
| **redis** | 消息队列 + AOF 持久化 | 2 CPU, 8GB | StatefulSet ×1 |
| **rtsp-ingest** | RTSP 录制, 10分钟分片 | 1 CPU, 2GB | 1 pod/摄像头 |
| **dispatcher** | 重叠窗口生成 (30s overlap) | 1 CPU, 2GB | 1 pod/摄像头 |
| **semantic-analyzer** | 唯一编排者: 抽帧→嵌入→聚类→VLM→结果 | 4 CPU, 8GB | HPA 2-6 |
| **embedding-service** | SigLIP 向量编码 | 4 CPU, 16GB, **1×T4** | HPA 1-3 |
| **vlm-proxy** | 并发阀门 (max=2) | 1 CPU, 1GB | 2 replicas |
| **vllm-server** | Qwen3-VL-32B-AWQ 推理 | 8 CPU, 64GB, **2×T4** | StatefulSet ×1 |
| **aggregator** | 结果去重 (±15s 窗口) | 1 CPU, 2GB | HPA 2-4 |

### GPU 分配

| GPU ID | 分配 | 显存占用 |
|--------|------|----------|
| GPU 0 | embedding-service | ~3 GB |
| GPU 1 | 预留扩展 | - |
| GPU 2-3 | vLLM Server (TP=2) | ~24 GB |
| GPU 4-7 | 预留扩展 | - |

---

## P0 稳定性措施

### 1. 编排统一 ✅
- **Semantic Analyzer** 作为唯一编排者
- 简化消息流: 仅 `video_tasks` 和 `result_tasks` 两个队列
- 避免多级消息传递导致的复杂性

### 2. 文件幂等 ✅
- **原子写入**: tmp 文件 → fsync → rename
- **确定性 job_id**: `{camera_id}_{seg_end_ts}`
- **去重检查**: 处理前检查文件是否已存在

### 3. 可观测 ✅
- **JSON 结构化日志**: 统一格式，含 trace_id/job_id/camera_id
- **trace_id 贯穿**: 从 API Gateway 生成，全链路透传
- **健康检查**: 所有 HTTP 服务提供 /health 端点

### 4. 可恢复 ✅
- **Redis AOF**: `appendfsync everysec`，数据持久化
- **Worker 自愈**: 处理 N 个任务后优雅退出，由 K8S 重启
- **K8S 探针**: liveness + readiness probe 自动重启异常 Pod

---

## 阶段1.5 实现

### 重叠窗口策略

```
RTSP 录制: 10分钟固定分片 (remux, 不转码)

seg[0]        seg[1]        seg[2]
├──────────┤  ├──────────┤  ├──────────┤
0         10  10        20  20        30  (分钟)

Dispatcher 生成重叠窗口:

window[1] = tail(seg[0], 30s) + seg[1]
          ├─────┤├──────────┤
          overlap   seg[1]
          9:30     10:00   20:00
```

### 去重策略

```python
# Aggregator 去重规则
if abs(event_time - cached_time) <= 15:  # ±15秒
    if new_confidence > cached_confidence:
        replace(cached_event, new_event)  # 保留高置信度
    else:
        discard(new_event)  # 丢弃重复
```

---

## 文件清单

### 微服务代码 (`workers/`)

| 文件 | 行数 | 描述 |
|------|------|------|
| `common/logging_config.py` | ~80 | JSON 日志格式化 |
| `common/redis_client.py` | ~150 | Redis Streams 客户端 |
| `rtsp_ingest.py` | ~160 | RTSP 录制 |
| `dispatcher.py` | ~250 | 重叠窗口生成 |
| `embedding_service.py` | ~230 | SigLIP HTTP 服务 |
| `vlm_proxy.py` | ~180 | VLM 并发控制 |
| `semantic_analyzer.py` | ~400 | 唯一编排者 |
| `aggregator.py` | ~280 | 结果去重 |
| `api_gateway.py` | ~200 | REST API |

### Docker 镜像 (`docker/`)

| 镜像 | 基础镜像 | GPU |
|------|---------|-----|
| traffic-vlm-base | nvidia/cuda:12.1.1 | - |
| api-gateway | traffic-vlm-base | - |
| embedding-service | traffic-vlm-base | ✅ |
| vlm-proxy | traffic-vlm-base | - |
| semantic-analyzer | traffic-vlm-base | - |
| rtsp-ingest | traffic-vlm-base | - |
| dispatcher | traffic-vlm-base | - |
| aggregator | traffic-vlm-base | - |

### K8S 清单 (`k8s/`)

| 文件 | 内容 |
|------|------|
| `00-namespace.yaml` | Namespace + PV/PVC |
| `01-redis.yaml` | Redis StatefulSet |
| `02-vllm-server.yaml` | vLLM StatefulSet (2×GPU) |
| `03-embedding-service.yaml` | Embedding Deployment (1×GPU) |
| `04-vlm-proxy.yaml` | VLM Proxy Deployment |
| `05-semantic-analyzer.yaml` | Semantic Analyzer + HPA |
| `06-aggregator.yaml` | Aggregator + HPA |
| `07-api-gateway.yaml` | API Gateway + NodePort |
| `08-rtsp-ingest.yaml` | RTSP Ingest + Dispatcher (示例) |

---

## 部署指南

### 快速部署

```bash
# 1. 一键部署
./scripts/deploy_remote.sh --remote r6500-g4-2 --user bj

# 2. 验证部署
./scripts/smoke_test.sh --remote r6500-g4-2

# 3. 故障测试
./scripts/chaos_test.sh --remote r6500-g4-2 --test all
```

### 手动部署

```bash
# 1. SSH 到远程服务器
ssh bj@r6500-g4-2

# 2. 同步代码
scp -r workers docker k8s scripts bj@r6500-g4-2:/data/app/

# 3. 构建镜像
cd /data/app && ./docker/build-images.sh

# 4. 部署 K8S
kubectl apply -f k8s/
```

### 验证命令

```bash
# 查看 Pod 状态
kubectl get pods -n traffic-vlm

# 查看日志
kubectl logs -f deployment/semantic-analyzer -n traffic-vlm

# 访问 API
curl http://r6500-g4-2:30500/health
curl http://r6500-g4-2:30500/api/v1/stats

# GPU 使用
kubectl exec -it statefulset/vllm-server -n traffic-vlm -- nvidia-smi
```

---

## 扩展指南

### 添加新摄像头

```yaml
# 复制 k8s/08-rtsp-ingest.yaml 中的模板
# 修改以下字段:
#   - name: rtsp-ingest-{camera_id}
#   - camera: "{camera_id}"
#   - --camera-id: "{camera_id}"
#   - --rtsp-url: "rtsp://..."
```

### 扩展 VLM 实例

```yaml
# 修改 k8s/02-vllm-server.yaml
spec:
  replicas: 2  # 增加实例数

# 需要更多 GPU，修改资源请求:
resources:
  limits:
    nvidia.com/gpu: "2"  # 每实例 2 GPU
```

### 切换 SigLIP 模型

```bash
# 更新 ConfigMap
kubectl create configmap embedding-config \
  --from-literal=MODEL_NAME=google/siglip2-xxx \
  -n traffic-vlm --dry-run=client -o yaml | kubectl apply -f -

# 滚动更新
kubectl rollout restart deployment/embedding-service -n traffic-vlm
```

---

## 已知限制

1. **RTSP 网络**: 需确保服务器可访问 172.21.x.x 网段
2. **模型下载**: Qwen3-VL-32B-AWQ (~20GB) 需 HuggingFace 访问
3. **单节点**: 当前设计为单节点 K8S，多节点需额外配置
4. **GPU 固定**: vLLM 固定使用 GPU 2-3，需手动调整 CUDA_VISIBLE_DEVICES

---

## 后续优化

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P1 | 添加 Prometheus + Grafana 监控 | 待做 |
| P1 | 实现实际 RTSP 流测试 | 待做 |
| P2 | Helm Chart 打包 | 待做 |
| P2 | 多节点集群支持 | 待做 |
| P3 | 边缘计算预处理 | 待做 |

---

## 附录

### A. 环境要求

- **服务器**: 8×T4 GPU, 370GB RAM, 3.4TB NVMe
- **系统**: Ubuntu 22.04
- **Docker**: 24.x+
- **K8S**: 1.29.x
- **CUDA**: 12.1+
- **Python**: 3.10+

### B. 端口映射

| 服务 | 内部端口 | 外部端口 |
|------|---------|---------|
| API Gateway | 5000 | 30500 (NodePort) |
| Embedding Service | 8080 | - |
| VLM Proxy | 8001 | - |
| vLLM Server | 8000 | - |
| Redis | 6379 | - |

### C. 联系方式

- 问题反馈: https://github.com/anthropics/claude-code/issues
