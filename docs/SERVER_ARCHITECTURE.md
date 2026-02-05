# 服务器端代码架构文档

> 基于 `/data/app/workers/` 目录实际代码内容，共 24 个 Python 文件，8171 行代码。
> 服务器：r6500-g4-2 (8×T4 GPU, K8S 单节点)

---

## 系统架构总览

```
RTSP 摄像头 (29路)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  数据采集层                                                       │
│  rtsp_ingest.py ──→ dispatcher.py                                │
│  (60s分片录制)       (重叠窗口生成)                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ XADD video_tasks
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Redis Streams 消息队列  (common/redis_client.py)                 │
│  ┌──────────────┐ ┌───────────────┐ ┌─────────────┐ ┌────────┐ │
│  │ video_tasks  │ │ result_tasks  │ │ edge_events │ │ dlq    │ │
│  └──────┬───────┘ └───────▲───────┘ └──────┬──────┘ └────────┘ │
└─────────┼─────────────────┼────────────────┼────────────────────┘
          │                 │                │
          ▼                 │                ▼
┌─────────────────────────────────────────────────────────────────┐
│  语义分析层 (4种分析器，共享基类 semantic_base.py)                    │
│                                                                   │
│  semantic_analyzer.py (事故检测 ×12 replicas)                      │
│  mv_violation_analyzer.py (机动车违法)                              │
│  ebike_v2_analyzer.py (二轮车违法)                                  │
│  ads_behavior_analyzer.py (自动驾驶行为)                            │
│                           │                                       │
│         HTTP调用           │           HTTP调用                     │
│  ┌──────┴──────┐          │     ┌──────┴──────┐                   │
│  │ Embedding   │          │     │ VLM Proxy   │                   │
│  │ Service     │          │     │ (并发阀门)   │                   │
│  │ (SigLIP)    │          │     │    ↓        │                   │
│  │ GPU 0-3     │          │     │  vLLM       │                   │
│  └─────────────┘          │     │ GPU 4-7     │                   │
│                           │     └─────────────┘                   │
│             XADD result_tasks                                     │
└───────────────────────────┼───────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  后处理层                                                         │
│  aggregator.py (去重聚合)  accident_clipper.py (事故片段剪辑)      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  API & 监控层                                                     │
│  api_gateway.py (REST API)  redis_streams_exporter.py (Prometheus)│
└─────────────────────────────────────────────────────────────────┘
```

---

## 模块详细说明

### 1. 数据采集层

#### `rtsp_ingest.py` (226行) — RTSP 视频录制器

- 从 RTSP 摄像头录制 **60 秒固定分片**
- 使用 `ffmpeg -c copy` remux（不转码，零 CPU 开销）
- 原子写入：先写 `.tmp` 文件，完成后 rename 为 `seg_{timestamp}.mp4`
- **自愈机制**：每处理 50 个分片后主动退出，由 K8S 自动重启
- 自动清理超过 `RTSP_RETAIN_HOURS`（默认 2 小时）的旧分片
- 当前部署：**20 个 rtsp-ingest 实例**（对应 20 路摄像头）

#### `dispatcher.py` (371行) — 分片调度器

- 监听 `rtsp_ingest` 产出的 `seg_*.mp4` 文件
- 生成 **重叠窗口**：`window[i] = tail(seg[i-1], 10s) + seg[i]`，确保事件不被分片边界切断
- 使用 TS 中间格式 concat 再 remux 回 MP4
- 生成确定性 `job_id = {camera_id}_{seg_end_ts}`
- 通过 `XADD video_tasks` 投递到 Redis 消息队列
- 当前部署：**20 个 dispatcher 实例**（与 rtsp-ingest 一一对应）

---

### 2. 消息队列层

#### `common/redis_client.py` (401行) — Redis Streams 客户端封装

核心数据结构：

| 数据类 | 用途 |
|--------|------|
| `VideoTask` | 视频任务消息（job_id, camera_id, window_path, analysis_type） |
| `ResultTask` | 分析结果消息（job_id, is_accident, confidence, processing_time_sec） |
| `EdgeEvent` | 边缘触发事件（camera_id, similarity_score, keyframes_base64） |

Redis Streams：

| Stream | 用途 | 上限 |
|--------|------|------|
| `video_tasks` | 视频分析任务队列 | 10000 条 |
| `result_tasks` | 分析结果队列 | 10000 条 |
| `edge_events` | 边缘端触发事件队列 | 10000 条 |
| `dlq_tasks` | 死信队列（处理失败的任务） | 1000 条 |

其他功能：
- 消费者组管理（`xgroup_create`, `xreadgroup`, `xack`）
- `task_status:{job_id}` Hash 存储任务状态（pending → processing → done/error）
- `emb:{video_id}` 嵌入向量缓存，避免重复计算

---

### 3. 语义分析层

#### `semantic_base.py` (845行) — 多语义分析基类（ABC 抽象类）

所有分析器的共享逻辑，子类只需实现模板和 Prompt。

**核心处理管道**：

```
1. extract_frames_with_mog2()   — MOG2 运动检测 + 自适应抽帧 (640×360@12fps)
2. compute_embeddings()          — HTTP 调用 Embedding Service 批量编码帧向量
3. compute_similarity_scores()   — 帧向量与事故模板向量的余弦相似度
4. cluster_frames_to_clips()     — 基于时间戳的滑动窗口聚类 (10s 窗口)
5. select_keyframes()            — 每个 Clip 选择 Top-N 关键帧
6. call_vlm()                    — HTTP 调用 VLM Proxy 进行视觉理解分析
7. save_result()                 — 结果序列化为 .result.json.gz 并 XADD result_tasks
```

抽象方法（子类必须实现）：

| 方法 | 说明 |
|------|------|
| `get_templates()` | 返回语义模板列表 |
| `get_vlm_prompt()` | 返回 VLM 分析 Prompt |
| `parse_vlm_response()` | 解析 VLM 响应为 AnalysisResult |

clip_score 阈值：只有 `clip_score >= 0.35` 的片段才送 VLM 分析。

#### `semantic_analyzer.py` (719行) — 事故检测分析器（核心主力）

- **独立实现，不继承 semantic_base**（历史遗留：此文件最早编写，semantic_base 是后来为多类型检测抽出的基类，但事故检测器从未迁移）
- 与 semantic_base 存在大量重复逻辑（抽帧、编码、相似度、聚类、VLM调用等），两边各维护一份
- 消费 `video_tasks` 队列，consumer group = `semantic_analyzers`
- 16 个中文事故模板（机动车碰撞、机非碰撞、人车碰撞等）
- VLM 事故判定 Prompt：`YES / NO / UNCERTAIN / POST_EVENT_ONLY`
- GPU 视频解码支持：`FFMPEG_NVDEC_ENABLED` + `FFMPEG_NVDEC_DEVICE`
- 当前部署：**12 个 replicas**（semantic-analyzer-g0~g3 各 3 个）

> **已知问题**：ebike_v2_analyzer / mv_violation_analyzer / ads_behavior_analyzer 虽然部署运行，但因 dispatcher 只生成 `analysis_type=accident` 的任务，这三个分析器实际上全部跳过所有任务，处于空转状态。

#### `mv_violation_analyzer.py` (221行) — 机动车违法检测

- 继承 `SemanticAnalyzerBase`
- 检测类型：闯红灯、违法变道、压线行驶、逆行、违法停车、不礼让行人
- 60+ 英文语义模板
- consumer group = `mv_violation_workers`

#### `ebike_v2_analyzer.py` (556行) — 二轮车违法检测 V2

- 继承 `SemanticAnalyzerBase`
- 检测类型：闯红灯、逆行、载人超员、不戴头盔、占用机动车道
- **V2 特性**：margin + Top-K 打分策略，Camera Cooldown Gate
- consumer group = `ebike_violation_workers`
- 配合 `ebike_v2_scorer.py` (489行) 实现三元组评分：`margin = pos - 0.7*hn - 0.3*neg`

#### `ebike_violation_analyzer.py` (223行) — 二轮车违法检测 V1（已停用）

- 70+ 英文语义模板，纯 Embedding + VLM
- 已被 ebike_v2_analyzer 替代，replicas=0

#### `ads_behavior_analyzer.py` (476行) — 自动驾驶行为检测

- 继承 `SemanticAnalyzerBase`
- 检测类型：示廓灯状态、自动驾驶模式异常行为
- **两阶段 VLM**：1) 检测示廓灯是否存在 → 2) 分析行为
- 50+ 英文语义模板
- consumer group = `ads_behavior_workers`

#### `semantic_analyzer_v2.py` (558行) — V2 增强版分析器

- 继承 `SemanticAnalyzerBase`
- 支持 YAML 模板定义 positives/hard_negatives/negatives
- 新评分公式：`group_score = weight * (pos_sim - β*hard_sim - γ*neg_sim)`
- 热重载支持
- 可替代 mv_violation / ebike_violation 的模板硬编码方式

---

### 4. AI 服务层

#### `embedding_service.py` (301行) — SigLIP 向量编码服务

- 加载模型：`google/siglip-base-patch16-384`
- Flask HTTP 服务，端口 8080
- API 接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/embeddings/images` | POST | 批量图像编码（支持 base64 和文件路径） |
| `/embeddings/text` | POST | 文本编码 |
| `/health` | GET | 健康检查 |

- GPU 加速：当前使用 GPU 0-3（4 张 T4）
- 批量处理：`BATCH_SIZE=32`
- 自愈：处理 1000 个请求后退出重启

#### `vlm_proxy.py` (224行) — VLM 并发阀门

- FastAPI + uvicorn，端口 8001
- **核心功能**：限制对 vLLM 的并发请求数为 2，防止洪峰
- 转发所有请求到 `VLLM_BASE_URL`（vLLM OpenAI API）
- 自动重试 2 次，超时 120 秒
- trace_id 透传
- 统计面板：`/stats`

#### vLLM（systemd 服务，非 K8S）

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-VL-32B-Instruct-AWQ |
| GPU | 4-7（4 张 T4，TP=4） |
| 端口 | 8000 |
| max-num-seqs | 6 |
| attention backend | TRITON_ATTN（T4 兼容） |

---

### 5. 后处理层

#### `aggregator.py` (400行) — 结果聚合 + 去重

- 消费 `result_tasks` 队列，consumer group = `aggregators`
- **去重逻辑**：同一 camera_id 在 ±15 秒内的多个结果合并为一个事件
- 保留 VLM confidence 最高的结果
- 幂等落盘：原子写入到 `/data1/aggregated/`
- 生成每日摘要（daily summary）
- 更新 `task_status` Hash
- 当前部署：**2 个 replicas**

#### `accident_clipper.py` (366行) — 事故视频剪辑器

- 监控 `/data1/results/` 目录的分析结果
- 检测到事故时自动：
  1. 剪辑事故视频片段（事故时间 ±30 秒）
  2. 提取关键帧截图
  3. 生成 Markdown 格式事故报告
- 存储到 `/data1/accidents/` 目录
- 摄像头信息映射（路口、方向、设备类型）

---

### 6. API & 监控层

#### `api_gateway.py` (499行) — REST API 网关

- Flask 应用，端口 30500
- 核心接口：

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/v1/tasks` | POST | 手动提交视频分析任务 |
| `/api/v1/tasks/<job_id>` | GET | 查询任务状态 |
| `/api/v1/tasks/<job_id>/result` | GET | 获取分析结果 |
| `/api/v1/cameras/<camera_id>/events` | GET | 查询摄像头事件 |
| `/api/v1/edge/events` | POST | 接收边缘触发事件 |
| `/api/v1/edge/events` | GET | 列出边缘触发事件 |
| `/api/v1/stats` | GET | 系统统计 |
| `/health` | GET | 健康检查 |

- 自动生成 `trace_id` 并透传到所有下游服务
- 当前部署：**2 个 replicas**

#### `redis_streams_exporter.py` (110行) — Prometheus 指标导出

- 端口 9122
- 导出指标：

| 指标 | 类型 | 说明 |
|------|------|------|
| `redis_stream_length` | Gauge | 队列长度 |
| `redis_stream_pending_messages` | Gauge | 未 ACK 消息数 |
| `redis_stream_consumer_lag` | Gauge | Consumer lag |
| `task_status_total` | Gauge | 按状态统计任务数（done/processing） |
| `result_total` | Gauge | 结果分布（事故/非事故） |
| `result_stream_length` | Gauge | 结果队列总条数 |
| `processing_time_avg_seconds` | Gauge | 平均处理时间（最近 200 条） |
| `processing_time_latest_seconds` | Gauge | 最新 Clip 处理时间 |
| `e2e_latency_avg_seconds` | Gauge | 平均端到端延迟 |
| `e2e_latency_latest_seconds` | Gauge | 最新端到端延迟 |

---

### 7. 边缘计算层（规划中）

#### `edge_trigger.py` (510行) — 边缘端事故预检测触发器

- 设计部署在边缘云（靠近摄像头端）
- 使用轻量 SigLIP 计算帧与事故模板的相似度
- 低阈值（0.12）保证高召回，超阈值时上报中心云
- 支持 CPU-only 模式（ONNX Runtime）
- 通过 `POST /api/v1/edge/events` 上报到 API Gateway
- 滑动窗口参数：60 秒窗口，5 秒滑动

---

### 8. 公共工具层

| 文件 | 行数 | 功能 |
|------|------|------|
| `common/redis_client.py` | 401 | Redis Streams 封装，3 条队列 + 状态管理 + 向量缓存 |
| `common/logging_config.py` | 103 | JSON 结构化日志，支持 trace_id/job_id/camera_id 链路追踪 |
| `common/gpu_image_utils.py` | 136 | PyTorch GPU 图片压缩（自动回退 CPU） |
| `common/image_compress.py` | 51 | FFmpeg 图片压缩（无 PIL 依赖） |
| `vlm_debug_logger.py` | 383 | VLM 调试日志系统，支持采样率控制、请求/响应记录 |
| `ebike_v2_scorer.py` | 489 | 三元组评分器：margin = pos - 0.7*hn - 0.3*neg |

---

## 当前部署拓扑

### K8S traffic-vlm 命名空间

| 组件 | 实例数 | 对应代码 |
|------|--------|----------|
| rtsp-ingest-* | 20 | `rtsp_ingest.py` |
| dispatcher-* | 20 | `dispatcher.py` |
| semantic-analyzer-g0~g3 | 12 (4×3) | `semantic_analyzer.py` |
| embedding-service | 1 | `embedding_service.py` |
| vlm-proxy | 1 | `vlm_proxy.py` |
| aggregator | 2 | `aggregator.py` |
| api-gateway | 2 | `api_gateway.py` |
| accident-clipper | 1 | `accident_clipper.py` |
| ads-behavior-analyzer | 1 | `ads_behavior_analyzer.py` |
| ebike-v2-analyzer | 1 | `ebike_v2_analyzer.py` |
| mv-violation-analyzer | 1 | `mv_violation_analyzer.py` |

### 监控命名空间

| 组件 | 实例数 | 用途 |
|------|--------|------|
| prometheus-grafana | 1 | 可视化监控面板 |
| prometheus-kube-prometheus-operator | 1 | Prometheus 运维 |
| redis-streams-exporter | 1 | Redis Streams 指标导出 |
| redis-exporter | 1 | Redis 通用指标导出 |

### Bare Metal 服务

| 组件 | 管理方式 | GPU |
|------|----------|-----|
| vLLM (Qwen3-VL-32B-AWQ) | systemd | GPU 4-7 (TP=4) |

### GPU 分配

| GPU | 用途 |
|-----|------|
| GPU 0-3 | Embedding Service (SigLIP) + NVDEC 视频解码 |
| GPU 4-7 | vLLM (Qwen3-VL-32B-AWQ, Tensor Parallel) |

---

## 数据流总结

```
摄像头 RTSP (29路)
  → rtsp_ingest.py (60s 录制，remux 不转码)
  → dispatcher.py (10s 重叠窗口拼接)
  → Redis video_tasks (XADD)
  → semantic_analyzer.py (×12 消费者)
    ├→ HTTP → embedding_service.py (SigLIP 帧编码, GPU 0-3)
    ├→ 相似度计算 → 时间聚类 → 关键帧选择
    ├→ clip_score >= 0.35 的片段:
    │   └→ HTTP → vlm_proxy.py (并发=2) → vLLM (Qwen3-VL, GPU 4-7)
    └→ Redis result_tasks (XADD)
  → aggregator.py (±15s 去重, ×2)
  → accident_clipper.py (事故片段 ±30s 剪辑)
  → /data1/aggregated/ (最终结果)
```

---

## 文件清单

| 文件路径 | 行数 | 功能概述 |
|----------|------|----------|
| `workers/rtsp_ingest.py` | 226 | RTSP 60s 分片录制 |
| `workers/dispatcher.py` | 371 | 重叠窗口生成 + 任务投递 |
| `workers/semantic_analyzer.py` | 719 | 事故检测主力分析器 |
| `workers/semantic_base.py` | 845 | 多语义分析基类 |
| `workers/semantic_analyzer_v2.py` | 558 | V2 YAML 模板分析器 |
| `workers/mv_violation_analyzer.py` | 221 | 机动车违法检测 |
| `workers/ebike_v2_analyzer.py` | 556 | 二轮车违法检测 V2 |
| `workers/ebike_v2_scorer.py` | 489 | 三元组评分器 |
| `workers/ebike_violation_analyzer.py` | 223 | 二轮车违法检测 V1（已停用） |
| `workers/ads_behavior_analyzer.py` | 476 | 自动驾驶行为检测 |
| `workers/embedding_service.py` | 301 | SigLIP 向量编码服务 |
| `workers/vlm_proxy.py` | 224 | VLM 并发控制代理 |
| `workers/aggregator.py` | 400 | 结果聚合去重 |
| `workers/accident_clipper.py` | 366 | 事故视频剪辑 |
| `workers/api_gateway.py` | 499 | REST API 网关 |
| `workers/edge_trigger.py` | 510 | 边缘端预检测触发器 |
| `workers/redis_streams_exporter.py` | 110 | Prometheus 指标导出 |
| `workers/vlm_debug_logger.py` | 383 | VLM 调试日志 |
| `workers/common/redis_client.py` | 401 | Redis Streams 客户端 |
| `workers/common/logging_config.py` | 103 | JSON 结构化日志 |
| `workers/common/gpu_image_utils.py` | 136 | GPU 图片压缩 |
| `workers/common/image_compress.py` | 51 | FFmpeg 图片压缩 |

**总计：24 个文件，8171 行代码**
