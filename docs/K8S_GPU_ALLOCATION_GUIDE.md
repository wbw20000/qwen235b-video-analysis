# K8S GPU 分配指南

> 基于 2026-01-29 Embedding Service 扩容踩坑经验总结

## 一、服务器 GPU 布局

### r6500-g4-2 (8x Tesla T4)

| GPU | 显存 | 当前使用者 | 分配方式 |
|-----|------|------------|----------|
| **0** | 15 GB | Embedding Service Pod 1 | CUDA_VISIBLE_DEVICES=0 |
| **1** | 15 GB | Embedding Service Pod 2 | CUDA_VISIBLE_DEVICES=1 |
| **2** | 15 GB | vLLM Worker TP0 | nohup 进程 |
| **3** | 15 GB | vLLM Worker TP1 | nohup 进程 |
| **4-7** | 15 GB x 4 | 空闲 | 可用于扩容 |

---

## 二、核心问题：K8S 与非 K8S 进程的 GPU 冲突

### 问题描述

vLLM 以 **nohup 进程**运行（非 K8S Pod），占用 GPU 2,3。但 K8S nvidia device plugin **不知道**这些 GPU 已被占用，会将新 Pod 分配到这些 GPU，导致 **CUDA OOM**。

### 错误现象

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 13.00 MiB is free.
Process 190902 has 14.02 GiB memory in use.
```

### 根因

```
K8S Pod 请求 nvidia.com/gpu: 1
    ↓
K8S nvidia device plugin 分配 GPU（可能是 2 或 3）
    ↓
该 GPU 已被 vLLM nohup 进程占用
    ↓
Pod 启动时 CUDA OOM → CrashLoopBackOff
```

---

## 三、解决方案

### 方案 A：移除 K8S GPU 资源请求 + 手动指定 GPU（推荐）

**原理**：不让 K8S 分配 GPU，而是通过 `CUDA_VISIBLE_DEVICES` 环境变量手动指定。

**步骤**：

1. **移除 GPU 资源请求**
```bash
kubectl patch deployment <name> -n traffic-vlm --type=json -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/resources/limits/nvidia.com~1gpu"},
  {"op": "remove", "path": "/spec/template/spec/containers/0/resources/requests/nvidia.com~1gpu"}
]'
```

2. **设置 CUDA_VISIBLE_DEVICES**
```bash
# 单 GPU 指定
kubectl set env deployment/<name> -n traffic-vlm CUDA_VISIBLE_DEVICES=0

# 多 GPU 可选（Pod 会选第一个）
kubectl set env deployment/<name> -n traffic-vlm CUDA_VISIBLE_DEVICES=0,1,4,5,6,7
```

### 方案 B：多 Deployment 隔离 GPU

当需要多个 Pod 使用不同 GPU 时，**必须创建多个 Deployment**，每个指定不同的 `CUDA_VISIBLE_DEVICES`。

**错误做法**：
```bash
# 单 Deployment 设置 CUDA_VISIBLE_DEVICES=0,1 并 replicas=2
# 结果：两个 Pod 都会选 cuda:0（物理 GPU 0），造成冲突
```

**正确做法**：
```bash
# Deployment 1: embedding-service
kubectl set env deployment/embedding-service -n traffic-vlm CUDA_VISIBLE_DEVICES=0

# Deployment 2: embedding-service-gpu1
kubectl set env deployment/embedding-service-gpu1 -n traffic-vlm CUDA_VISIBLE_DEVICES=1
```

**创建第二个 Deployment**：
```bash
# 从现有 Deployment 复制并修改名称
kubectl get deployment embedding-service -n traffic-vlm -o yaml | \
  grep -v "resourceVersion\|uid\|creationTimestamp\|generation" | \
  sed "s/name: embedding-service$/name: embedding-service-gpu1/" | \
  kubectl apply -f -

# 设置不同的 GPU
kubectl set env deployment/embedding-service-gpu1 -n traffic-vlm CUDA_VISIBLE_DEVICES=1
```

**确保 Service 能选中所有 Pod**：
- 两个 Deployment 的 Pod 标签必须匹配 Service selector
- 默认复制时标签会改变，需要修正：
```bash
# 检查 Service selector
kubectl get svc embedding-service -n traffic-vlm -o jsonpath='{.spec.selector}'
# {"app":"embedding-service"}

# 确保两个 Deployment 的 Pod 都有 app=embedding-service 标签
```

---

## 四、常见错误与排查

### 错误 1：CUDA OOM（GPU 显存不足）

**症状**：
```
torch.OutOfMemoryError: CUDA out of memory
```

**排查**：
```bash
# 查看 GPU 占用
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv

# 对照 GPU UUID 找到物理 GPU 编号
nvidia-smi --query-gpu=index,uuid,memory.used --format=csv
```

**解决**：指定未被占用的 GPU

### 错误 2：OOMKilled（系统内存不足）

**症状**：
```
kubectl get pods → OOMKilled
Exit Code: 137
```

**排查**：
```bash
kubectl describe pod <name> -n traffic-vlm | grep -A5 "Last State:"
```

**解决**：增加内存限制
```bash
kubectl patch deployment <name> -n traffic-vlm --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "24Gi"}
]'
```

### 错误 3：两个 Pod 使用同一 GPU

**症状**：
```bash
nvidia-smi --query-compute-apps=...
# 两个 python 进程在同一 GPU UUID
```

**原因**：`CUDA_VISIBLE_DEVICES=0,1` 让两个 Pod 都能看到 GPU 0,1，但代码默认用 `cuda:0`（即物理 GPU 0）

**解决**：使用多 Deployment，每个指定单一 GPU

---

## 五、GPU 分配检查清单

### 扩容前检查

- [ ] 运行 `nvidia-smi` 确认哪些 GPU 已被占用
- [ ] 确认 vLLM 占用的 GPU 编号（通常是 2,3）
- [ ] 选择空闲 GPU 用于新 Pod

### 创建 Deployment 时

- [ ] 移除 `nvidia.com/gpu` 资源请求（避免 K8S 随机分配）
- [ ] 设置 `CUDA_VISIBLE_DEVICES` 指定单一 GPU
- [ ] 如需多 Pod，创建多个 Deployment，每个指定不同 GPU

### 扩容后验证

- [ ] `kubectl get pods` 确认所有 Pod 1/1 Running
- [ ] `nvidia-smi` 确认每个 GPU 只有一个进程
- [ ] `kubectl get endpoints` 确认所有 Pod 都在 Service 中

---

## 六、快速命令参考

```bash
# 查看 GPU 使用
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 查看 GPU 进程
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv

# 查看 Deployment 的 GPU 环境变量
kubectl get deployment <name> -n traffic-vlm -o jsonpath='{.spec.template.spec.containers[0].env}' | grep CUDA

# 设置 GPU
kubectl set env deployment/<name> -n traffic-vlm CUDA_VISIBLE_DEVICES=<gpu_id>

# 查看 Pod 状态
kubectl get pods -n traffic-vlm -l app=<app_label>

# 查看 Service Endpoints
kubectl get endpoints -n traffic-vlm <service_name>
```

---

## 七、当前生产配置

### Embedding Service (2 Pod)

| Deployment | GPU | CUDA_VISIBLE_DEVICES | 内存限制 |
|------------|-----|---------------------|----------|
| embedding-service | 0 | 0 | 24Gi |
| embedding-service-gpu1 | 1 | 1 | 24Gi |

### vLLM (nohup 进程)

| 进程 | GPU | 启动方式 |
|------|-----|----------|
| VLLM::Worker_TP0 | 2 | CUDA_VISIBLE_DEVICES=2,3 nohup |
| VLLM::Worker_TP1 | 3 | 同上（tensor parallel） |

### 可用 GPU

| GPU | 用途建议 |
|-----|----------|
| 4 | 可扩容第 3 个 Embedding Pod |
| 5 | 可扩容第 4 个 Embedding Pod |
| 6-7 | 预留 / 未来 vLLM 扩容 |

---

*文档更新: 2026-01-29*
*作者: Claude Code*
