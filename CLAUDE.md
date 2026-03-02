# 交通事故视频分析系统 - TrafficVLM

## 项目概述

基于 Qwen3-VL 大模型的交通事故智能检测系统，通过多阶段渐进式分析策略，实现高召回率、低误报率的事故检测。

**最新评测指标 (256视频测试)**:
| 指标 | 数值 |
|------|------|
| Accuracy | 93.4% |
| Precision | 99.4% |
| Recall | 91.7% |
| F1 Score | 95.4% |
| FPR | 1.6% |

## 项目结构

```
qwen235b/
├── app.py                      # Flask Web应用入口
├── templates/                  # 前端模板
│   ├── index.html             # 主页面（视频上传分析）
│   └── history.html           # 历史视频回溯分析页面
│
├── traffic_vlm/               # 核心分析引擎
│   ├── pipeline.py            # 主处理管道（端到端）
│   ├── config.py              # 配置中心（20+ dataclass）
│   ├── vlm_client.py          # VLM API客户端（Qwen3-VL）
│   ├── detector_and_tracker.py # YOLO检测 + ByteTrack跟踪
│   ├── clip_sampler.py        # FFmpeg视频剪辑
│   ├── embedding_indexer.py   # SigLIP向量检索
│   ├── temporal_clusterer.py  # 时间聚类
│   ├── coverage_scorer.py     # 覆盖率评分
│   ├── keyframe_selector.py   # 关键帧选择
│   ├── trajectory_scorer.py   # 轨迹碰撞评分
│   ├── vlm_sampling.py        # VLM高频抽帧
│   ├── tsingcloud_api.py      # 云控智行API客户端
│   ├── history_video_processor.py # 历史视频处理
│   ├── batch_processor.py     # 批量遍历处理器
│   ├── gpu_service.py         # GPU加速服务
│   └── visual_annotator.py    # 标注可视化
│
├── evaluation/                # 评测框架
│   ├── evaluator.py           # 评测核心（predict_file）
│   └── metrics.py             # 指标计算（TP/FP/TN/FN）
│
├── reporting/                 # 报告生成
│   ├── report_builder.py      # 诊断报告构建器
│   └── scorecard.py           # 评分卡
│
├── tools/                     # 工具脚本
│   ├── run_eval_to_output.py  # 评测运行器
│   ├── run_regression_eval.py # 回归测试
│   ├── threshold_sweep.py     # 阈值扫描
│   └── analysis/              # 分析工具
│
├── data/                      # 运行时数据
│   ├── video_results/         # 分析结果缓存（.result.json.gz）
│   ├── vlm_logs/              # VLM请求日志
│   └── index.db               # SQLite索引
│
├── uploads/                   # 上传视频存储
├── outputs/                   # 评测输出
└── venv/                      # Python虚拟环境
```

## 核心模块说明

### 1. Pipeline 主管道 (`traffic_vlm/pipeline.py`)

端到端处理流程：
```
视频输入 → 运动检测 → YOLO检测 → SigLIP检索 → 时间聚类 → 片段裁剪 → VLM分析 → 结果输出
```

### 2. VLM 客户端 (`traffic_vlm/vlm_client.py`)

- **支持模型**: qwen3-vl-plus (默认), qwen3-vl-32b, qwen3-vl-235b
- **三阶段分析**: S1快速(12帧) → S2升级(18帧) → S3困难场景(16帧)
- **四态判定**: YES / NO / UNCERTAIN / POST_EVENT_ONLY

### 3. 配置中心 (`traffic_vlm/config.py`)

关键配置类:
| 配置类 | 用途 |
|--------|------|
| `VLMConfig` | VLM调用参数（模型、帧数、阈值） |
| `DetectorConfig` | YOLO检测参数（置信度、模型路径） |
| `ProgressiveVLMConfig` | 渐进式VLM策略（S1/S2帧数） |
| `Stage3Config` | S3困难场景配置（天气关键词） |
| `CoverageConfig` | 覆盖率评分（pre_roll/post_roll） |
| `TsingcloudConfig` | 云控智行API凭据 |
| `BatchProcessConfig` | 批量处理配置 |

### 4. 评测框架 (`evaluation/evaluator.py`)

- `predict_file()`: 单视频预测
- 输出: TP/FP/TN/FN, Recall, FPR, Precision, F1

---

## 开发规范

### 虚拟环境使用

**强制要求**: 本项目所有代码执行必须在虚拟环境中进行。

```bash
# 运行代码
d:/project2025/qwen235b/venv/Scripts/python.exe xxx.py

# 安装依赖
d:/project2025/qwen235b/venv/Scripts/pip.exe install package-name
```

**禁止**: 直接使用 `python` 命令或系统Python路径

### 日志输出规范

- 长时间运行的任务使用后台模式（`run_in_background`）
- 避免一次性输出大量日志
- 优先查看最终统计结果而非逐条日志

### 评测输出规范

每次回归测试必须生成:

| 文件 | 格式 | 内容 |
|------|------|------|
| `summary.json` | JSON | 汇总指标 |
| `per_file.json` | JSON | 每视频结果 |
| `casebook.md` | Markdown | FN/FP案例分析 |
| `decision_trace.json` | JSON | 决策链追踪 |

### 评测命令

```bash
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/run_xxx --dump-video-results
```

### SSH 远程执行注意事项

**教训 (2026-02-04)**: SSH 后台执行 + 环境变量导出 = 容易失败

**错误做法**:
```bash
# 环境变量不会传递给 nohup 子进程！
ssh user@host "/path/to/script.sh &"
```

**正确做法**:
```bash
# 让脚本本身处理后台化，SSH 等待脚本初始化完成
ssh user@host "bash /path/to/script.sh"
# 脚本内部用 nohup ... & 实现后台运行
```

**原因**: 通过 SSH 非交互式后台执行时，脚本中 `export` 的环境变量可能**无法传递给 nohup 子进程**，导致 `CUDA_VISIBLE_DEVICES` 等关键配置失效。

**vLLM 启动脚本示例** (`/data/app/start_vllm.sh`):
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7  # 在脚本内设置
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
nohup python -m vllm.entrypoints.openai.api_server ... &  # 脚本内后台化
echo "vLLM started"  # 脚本返回
```

---

## General Rules

### Task Resumption

恢复或继续长时间运行的任务（评测、构建、实验）时，**必须先检查是否有缓存结果、检查点或部分进度**，避免从头重跑。不确定时询问用户。

- 评测任务：检查 `data/video_results/` 下的 `.result.json.gz` 缓存
- Docker 构建：检查镜像是否已存在（`docker images | grep tag`）
- 实验运行：检查 `outputs/` 目录下是否有之前的 `summary.json`

### Estimation & Analysis

估算计算资源或成本时：

1. **先确认架构模式**：批处理/事件驱动 vs 实时逐路推理，不同模式的资源需求差异巨大
2. **基于实际测试数据**：使用真实的评测报告数据（如 `summary.json` 中的处理时间）作为估算基础
3. **不要假设实时逐设备推理**：除非明确说明，本项目是事件驱动架构（Redis 队列触发分析），不是每路摄像头持续推理

---

## Infrastructure Context

本项目使用 vLLM + Kubernetes 架构进行视频/交通分析，GPU pods 分布在 8 张 T4 上。

### 关键架构细节

- **Redis 队列区分**：`video_tasks`（预处理任务）和 `accident_tasks`（分析任务）是不同队列，注意区分
- **vLLM 服务管理**：可能同时存在 systemd service 和 legacy 启动脚本（`start_vllm.sh`），操作前**两者都要检查**
- **模型权重分离**：Docker 镜像**不应包含模型权重**，通过 volume mount 挂载（`/data/models/`）
- **使用本地 vLLM 端点**：K8S 内部通过 `llm-server:8000` 访问，**不使用 DashScope API**
- **HPA 注意事项**：GPU 工作负载的 HPA 应考虑 GPU 显存，而非仅 CPU 利用率

---

## Remote Server Operations

通过 SSH 操作远程服务器时：

- **连接超时**：设置至少 30s（`ssh -o ConnectTimeout=30`）
- **Tailscale VPN**：预期网络间歇性中断，长时间运行的远程命令需要内置重试逻辑
- **使用 tmux/screen**：长时间运行的任务（构建、部署）应在 tmux 会话中执行，防止 SSH 断连导致任务中断
- **Docker 上下文**：操作前验证 Docker context 和 containerd 镜像兼容性（`docker context ls`）
- **文件传输**：大文件传输优先使用 `rsync --partial --progress`，支持断点续传

---

## Docker Builds

Docker 镜像构建规范：

- **使用 BuildKit 缓存**：`DOCKER_BUILDKIT=1 docker build ...`，利用 `--mount=type=cache` 加速 pip 安装
- **Python 依赖管理**：安装 `uv` 替代 pip，显著加速依赖安装（`pip install uv && uv pip install -r requirements.txt`）
- **模型权重分离**：镜像内**不打包模型文件**，通过 K8S hostPath 或 PV 挂载 `/data/models/`
- **构建前检查磁盘空间**：`df -h` 确认目标驱动器有足够空间（镜像构建常需 5-10 GB 临时空间）
- **Docker 数据根目录**：不要假设在 C: 盘（Windows）或 `/var/lib/docker`（Linux），检查 `docker info | grep "Docker Root Dir"`
- **构建时使用 `--no-cache`**：当代码层可能被缓存旧内容时（特别是 `COPY workers/` 层）

---

## 关键参数设定

| 参数 | 值 | 理由 |
|------|------|------|
| VLM模型 | qwen3-vl-plus | 性价比最优，API调用成本低 |
| YOLO模型 | yolo11s.pt | 检测精度与速度平衡 |
| 检测置信度 | 0.2 | 低阈值确保不漏检 |
| 检测尺寸 | 1280 | 远距离目标检测 |
| clip_score_threshold | 0.35 | 平衡召回与误报 |
| S1帧数 | 12 | 快速判定足够帧数 |
| S2帧数 | 18 | 升级分析增加帧数 |
| SigLIP模型 | siglip-base-patch16-384 | 语义检索精度高 |

## 远程服务器

### r6500-g4-2 (GPU服务器)

#### 基本信息
| 项目 | 值 |
|------|-----|
| 主机名 | r6500-g4-2 |
| IP | 100.123.59.56 (Tailscale) |
| 用户名 | bj |
| SSH密钥 | ~/.ssh/id_ed25519 |
| sudo密码 | 1q2w3e |
| 系统 | Ubuntu 22.04 (内核 6.8.0-90-generic) |

#### CPU
| 项目 | 配置 |
|------|------|
| 型号 | Intel Xeon Gold 5218R @ 2.10GHz |
| 插槽 | 2 颗 |
| 核心/线程 | 40核 / 80线程 |

#### 显卡
| 项目 | 配置 |
|------|------|
| 型号 | NVIDIA Tesla T4 |
| 数量 | 8 张 |
| 单卡显存 | 15 GB |
| 总显存 | 120 GB |
| 驱动版本 | 590.48.01 |
| CUDA 版本 | 13.1 |

#### 内存
| 项目 | 配置 |
|------|------|
| 总容量 | 376 GB |
| 类型 | DDR4 ECC |
| 速度 | 3200 MT/s (运行 2666 MT/s) |
| 配置 | 32GB × 12条 |

#### 硬盘
| 设备 | 容量 | 类型 | 挂载点 |
|------|------|------|--------|
| sda (三星 SSD) | 447 GB | SATA SSD | `/` (系统) |
| sdb (三星 SSD) | 447 GB | SATA SSD | 未挂载 |
| nvme0n1 (Intel P5520) | 1.7 TB | NVMe | `/data1` |
| nvme1n1 (Intel P5520) | 1.7 TB | NVMe | `/data` |
| **总存储** | **4.3 TB** | | |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端 | Flask, Python 3.10 |
| 视觉模型 | Qwen3-VL (阿里云DashScope API) |
| 目标检测 | YOLO11s + ByteTrack |
| 向量检索 | SigLIP (google/siglip-base-patch16-384) |
| 视频处理 | FFmpeg, OpenCV, decord |
| GPU加速 | PyTorch + CUDA 12.1 |
| 数据存储 | SQLite |

---

## 已完成：No-SigLIP 模型对比实验（2026-02-12）

### 实验概要

5 个 VLM 模型 x 261 视频（197事故 + 64非事故），跳过 SigLIP（hardcode similarity=1.0, template_hit=True），其他预处理组件全开。

### STRICT 模式结果

| 模型 | TP | FP | TN | FN | UNC | Recall | FPR | Precision | F1 | abstain% |
|------|----|----|----|----|-----|--------|-----|-----------|------|----------|
| A: 8B Instruct | 182 | 0 | 9 | 4 | 66 | 97.8% | 0.0% | 100.0% | 98.9% | 25.3% |
| B: 8B Thinking | 128 | 19 | 33 | 46 | 35 | 73.6% | 36.5% | 87.1% | 79.8% | 13.4% |
| **C: 32B Instruct** | **177** | **3** | **60** | **18** | **3** | **90.8%** | **4.8%** | **98.3%** | **94.4%** | **1.1%** |
| D: 32B Thinking | 160 | 2 | 56 | 29 | 14 | 84.7% | 3.4% | 98.8% | 91.2% | 5.4% |
| E: 32B-AWQ | 168 | 1 | 62 | 29 | 1 | 85.3% | 1.6% | 99.4% | 91.8% | 0.4% |

### 关键结论

1. **32B Instruct 综合最优**（F1=94.4%，两种模式稳定）
2. **AWQ 4bit 量化精度损失可控**（F1 降 2.6%，FPR 反而更低）
3. **Thinking 模型不适合分类任务**（犹豫多，不稳定）
4. **8B 系列模型能力不足**（大量弃权或幻觉）

### 输出目录

- 报告: `outputs/ablation_no_siglip_model_comparison.md`
- 各模型结果: `outputs/ablation_no_siglip_{model}/eval/summary.json`

---

## 进行中：预处理组件消融实验 v2（2026-02-13 开始）

### 核心问题

MOG2、YOLO+ByteTrack、智能选帧、元数据文本等预处理组件对最终 VLM 准确率的实际贡献是多少？哪些可以裁剪以降低算力成本？

### 管线数据流（代码追踪确认）

```
Video → MOG2运动检测 → 关键帧提取 → SigLIP语义检索 → 轨迹评分(默认OFF)
  → 时间聚类 → Clip剪辑 → YOLO+ByteTrack → KeyframeSelector智能选帧
  → MetadataPack元数据文本 → VLM S1/S2
```

**VLM 实际接收的输入**（Progressive模式）：
1. **原始帧**（无YOLO框）— 由 KeyframeSelector 信号驱动选出
2. **结构化文本 metadata_text** — 每帧检测目标(ID/类别/bbox)、最近距离、轨迹摘要
3. **tracks_text** — ID→类别映射表
4. **traffic_light_text** — 信号灯状态

### 实验条件设计（6 条件）

| 代号 | 名称 | MOG2 | SigLIP | YOLO(clip内) | 帧选择 | 元数据文本 | 回答的问题 |
|------|------|------|--------|-------------|--------|-----------|-----------|
| **C0** | Full Pipeline (基线) | ON | ON | ON | Smart | ON | 真正的基线 |
| **C1** | No-SigLIP | ON | **OFF** | ON | Smart | ON | SigLIP 语义过滤贡献 |
| **C2** | No-Metadata | ON | ON | ON | Smart | **OFF** | 元数据文本帮助还是干扰VLM |
| **C3** | No-YOLO | ON | ON | **OFF** | **Uniform** | OFF | YOLO+ByteTrack 整体贡献 |
| **C4** | No-SmartSelect | ON | ON | ON | **Uniform** | ON | 智能选帧 vs 均匀选帧 |
| **C6** | Minimal | **OFF** | **OFF** | **OFF** | **Uniform** | OFF | 全部预处理的总价值 |

### 条件间对比矩阵

| 对比 | 隔离变量 | 洞察 |
|------|---------|------|
| C0 vs C1 | SigLIP | 语义过滤是否提升准确率 |
| C0 vs C2 | Metadata text | 文本元数据对VLM判断的贡献（**最重要**） |
| C0 vs C3 | YOLO整体 | 目标检测+跟踪+选帧+元数据的整体贡献 |
| C0 vs C4 | Smart select | 信号驱动选帧 vs 简单均匀选帧 |
| C2 vs C3 | YOLO选帧(无metadata) | 分离"选帧贡献" vs "metadata贡献" |
| C0 vs C6 | 全部预处理 | 预处理管线的总体价值上限 |

### 数据集（644 视频，比上轮 261 扩大 147%）

| 类别 | 来源 | 数量 | 子计 |
|------|------|------|------|
| 事故（正样本） | 根目录 mp4 | 192 | |
| | 01-自动驾驶车路侧/ | 6 | |
| | 车网路测事故/ | 16 | **214** |
| 非事故（负样本） | 1xxx 原有 | 28 | |
| | 2xxx 新增路测 | 36 | |
| | fp_ 实际车网路测 | 337 | |
| | 非机动车违法事件（从正样本移入） | 29 | **430** |

**正负比 1:2.0**（上轮 3:1），负样本翻倍，FPR 测试力度大幅提升。

**数据集路径**：
- 事故目录: `D:/project2025/qwen235b/uploads/大样本事故数据集`（递归扫描，排除"非机动车违法事件视频"子目录）
- 非事故目录: `D:/project2025/qwen235b/uploads/大样本非交通事故数据集`
- 额外非事故: `D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频`

### 模型策略 — 两阶段

**第一阶段**（2 模型 x 6 条件 = 12 runs）：
| 模型 | API 端点 | 理由 |
|------|---------|------|
| **32B Instruct** | DashScope 云端 | 上轮最佳(F1=94.4%) |
| **32B-AWQ** | 远程 vLLM (100.123.59.56:8000) | 当前生产模型(F1=91.8%) |

**第二阶段**：根据第一阶段结果，选差异最大的 3 个条件，用 8B-Instruct / 32B-Thinking / 8B-Thinking 验证。

### 缓存复用策略

| 预处理组 | 包含条件 | 说明 |
|---------|---------|------|
| **组A** | C0, C2, C4 | 完整预处理，C2/C4 只改 VLM 调用方式 |
| **组B** | C1 | No-SigLIP 预处理（候选帧不同） |
| **组C** | C3 | No-YOLO 预处理（均匀选帧） |
| **组D** | C6 | Minimal（跳过 MOG2+SigLIP+YOLO） |

### 串行批量执行策略

**不使用并发**（Windows 内存限制），全部 12 runs 串行执行。

**批量脚本**: `tools/run_ablation_v2_batch.py`

```bash
# 启动全量执行（无需 Claude 会话在线）
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_ablation_v2_batch.py

# 仅查看进度汇总
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_ablation_v2_batch.py --summary-only

# Dry-run 模式（打印命令不执行）
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_ablation_v2_batch.py --dry-run

# 从指定 run 开始
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_ablation_v2_batch.py --start-from 3
```

**执行顺序**（含缓存依赖）：

```
Run  1: C0 instruct  ← 生成组A缓存
Run  2: C0 awq       ← 复用 Run1 缓存
Run  3: C2 instruct  ← 复用 Run1 缓存 + skip-metadata
Run  4: C2 awq       ← 复用 Run1 缓存 + skip-metadata
Run  5: C4 instruct  ← 复用 Run1 缓存 + force-uniform
Run  6: C4 awq       ← 复用 Run1 缓存 + force-uniform
Run  7: C1 instruct  ← 独立预处理 (no-siglip)
Run  8: C1 awq       ← 复用 Run7 缓存
Run  9: C3 instruct  ← 独立预处理 (no-yolo)
Run 10: C3 awq       ← 复用 Run9 缓存
Run 11: C6 instruct  ← 独立预处理 (minimal)
Run 12: C6 awq       ← 复用 Run11 缓存
```

**双层缓存保障**：
1. **Run 级跳过**：检查 `eval/summary.json` 是否已存在 → 整个 run 跳过
2. **视频级缓存**：evaluator 内部 `_try_load_cached_result()` 跳过已有 `.result.json.gz` 的视频
3. **预处理缓存复用**：`--reuse-preprocess` 参数跳过同组重复预处理

**中断恢复**：脚本中断后重启，自动跳过已完成 runs + 从视频级缓存断点继续。

### 输出目录规划

每条件每模型独立目录，互不覆盖：
```
outputs/ablation_v2_C0_32b_instruct/     # C0 基线, DashScope
outputs/ablation_v2_C0_32b_awq/          # C0 基线, vLLM
outputs/ablation_v2_C1_no_siglip_32b_instruct/
outputs/ablation_v2_C1_no_siglip_32b_awq/
outputs/ablation_v2_C2_no_metadata_32b_instruct/
outputs/ablation_v2_C2_no_metadata_32b_awq/
outputs/ablation_v2_C3_no_yolo_32b_instruct/
outputs/ablation_v2_C3_no_yolo_32b_awq/
outputs/ablation_v2_C4_uniform_32b_instruct/
outputs/ablation_v2_C4_uniform_32b_awq/
outputs/ablation_v2_C6_minimal_32b_instruct/
outputs/ablation_v2_C6_minimal_32b_awq/
```

### 代码变更清单

| 文件 | 变更 |
|------|------|
| `traffic_vlm/config.py` | 新增 `ablation_skip_mog2`, `ablation_force_uniform_frames` |
| `tools/run_eval_to_output.py` | 新增 CLI 参数 + 数据集排除/额外目录参数 |
| `evaluation/evaluator.py` | 配置透传 + `_scan_mp4()` 递归 + 排除逻辑 |
| `traffic_vlm/pipeline.py` | skip_mog2 / force_uniform / disable_yolo 代码路径 |

### 评测命令示例

```bash
# C0 基线 (32B Instruct)
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_v2_C0_32b_instruct \
  --model qwen3-vl-32b-instruct \
  --acc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集" \
  --nonacc-dir "D:/project2025/qwen235b/uploads/大样本非交通事故数据集" \
  --acc-exclude-subdir "非机动车违法事件视频" \
  --extra-nonacc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频" \
  --dump-video-results

# C2 No-Metadata (32B Instruct, 复用C0缓存)
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_v2_C2_no_metadata_32b_instruct \
  --model qwen3-vl-32b-instruct \
  --reuse-preprocess-dir outputs/ablation_v2_C0_32b_instruct/data \
  --ablation-skip-metadata \
  --acc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集" \
  --nonacc-dir "D:/project2025/qwen235b/uploads/大样本非交通事故数据集" \
  --acc-exclude-subdir "非机动车违法事件视频" \
  --extra-nonacc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频" \
  --dump-video-results

# C6 Minimal (32B AWQ, 远程vLLM)
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_v2_C6_minimal_32b_awq \
  --model qwen3-vl-32b \
  --ablation-skip-mog2 --ablation-skip-siglip --ablation-disable-yolo --ablation-force-uniform \
  --acc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集" \
  --nonacc-dir "D:/project2025/qwen235b/uploads/大样本非交通事故数据集" \
  --acc-exclude-subdir "非机动车违法事件视频" \
  --extra-nonacc-dir "D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频" \
  --dump-video-results
```

### 预期假设

| 假设 | 预测 | 验证对比 |
|------|------|---------|
| H1: SigLIP 贡献有限 | Delta-F1 < 2% | C0 vs C1 |
| H2: Metadata 文本可能干扰 VLM | C2 的 F1 >= C0 | C0 vs C2 |
| H3: YOLO 主要贡献在选帧而非 metadata | C4 下降 > C2 下降 | C2 vs C4 |
| H4: Minimal 条件 F1 仍 >85% | VLM 视觉能力很强 | C6 结果 |
| H5: MOG2+SigLIP 对 FPR 有正贡献 | C6 FPR > C0 FPR | C0 vs C6 |

---

## 待办：跟踪器升级（解决 ID 跳变和轨迹断裂）

### 背景问题

当前使用 ByteTrack 进行多目标跟踪，存在以下问题：
- **ID 跳变**：碰撞后目标变形导致 ID 变化
- **轨迹断裂**：遮挡、人车分离等场景下 track_id 不连续
- **影响 VLM**：断裂的轨迹可能误导 VLM 的事故判断

### 升级方案（分阶段）

#### 第 1 阶段：低改造、快速验证（1-2 天）

| 跟踪器 | 特点 | 适用场景 |
|--------|------|----------|
| **BoT-SORT-ReID** | 最像 ByteTrack 的升级版，MOT17 工程派最强 | 直接替换 ByteTrack |
| **StrongSORT++** | 对断轨后的补链/平滑更友好 | DanceTrack/MOT17 |
| **Hybrid-SORT** | DanceTrack 上明显强于 ByteTrack | 抗交叉遮挡 |

#### 第 2 阶段：学习式多帧上下文关联

| 跟踪器 | 特点 | 适用场景 |
|--------|------|----------|
| **MOTRv2** | DanceTrack 提升巨大，Transformer tracking-by-query | 最稳的落地起点 |
| **MOTIP / ColTrack** | DanceTrack 排名靠前 | 需评估工程化成熟度 |

#### 第 3 阶段：最强关联（算力/链路更重）

| 跟踪器 | 特点 | 适用场景 |
|--------|------|----------|
| **SAM2MOT** | 对 ID 连续性提升最大，强调 zero-shot | 追求最不容易误导 VLM |

### 相关文件

- `traffic_vlm/detector_and_tracker.py`: 当前 YOLO + ByteTrack 实现
- `traffic_vlm/custom_bytetrack.yaml`: ByteTrack 配置
- `traffic_vlm/trajectory_scorer.py`: 依赖跟踪结果的轨迹评分

---

## 待办（最低优先级）：FN 案例分析与改进方向

### 测试结果概述 (261 样本)

| 指标 | 数值 |
|------|------|
| Recall | 88.6% |
| Precision | 100% |
| FN | 26 例 |

### FN 案例分类（26例）

| 类别 | 数量 | 说明 |
|------|------|------|
| POST_EVENT_ONLY 未计为事故 | 6 | 仅看到后果，未判为事故 |
| 困难场景漏检 | 5 | 夜间/雨天/远距离等 |
| VLM 高置信误判 | 11 | VLM 高置信度判为 NO |
| UNCERTAIN 未升级 | 4 | S1=UNCERTAIN 但未触发 S2 |

### 改进方向（优先级排序）

1. **POST_EVENT_ONLY 处理**（+3% Recall 潜力）
   - 将 POST_EVENT_ONLY 视为事故阳性
   - 修改 `evaluation/evaluator.py` 中的判定逻辑

2. **启用 S3 阶段**
   - `Stage3Config.enabled = True`
   - 针对困难场景（夜间/雨天）二次分析

3. **UNCERTAIN 升级机制**
   - S1=UNCERTAIN 时强制触发 S2
   - 当前 S2 触发条件可能遗漏部分 UNCERTAIN

4. **VLM 误判分析**
   - 需逐案例分析原因
   - 可能需要优化 prompt 或增加帧数

### 备注

性能提升空间有限（约 3-5%），属于边际改进，优先级最低。

---

## 远程部署 (r6500-g4-2 K8S)

### 当前部署状态 (2026-01-26)

| 组件 | 状态 | 部署方式 |
|------|------|----------|
| K8S 集群 | ✅ 运行中 | kubeadm 单节点 |
| NVIDIA Device Plugin | ✅ 8 GPU 就绪 | DaemonSet |
| Redis | ✅ 运行中 | K8S StatefulSet |
| vLLM (Qwen3-VL-32B-AWQ) | ✅ 运行中 | nohup (端口 8000) |
| Embedding Service | 待部署 | K8S Deployment |
| API Gateway | 待部署 | K8S Deployment |

### 关键路径

| 项目 | 路径 |
|------|------|
| vLLM 日志 | `/data/logs/vllm.log` |
| 模型文件 | `/data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ/` |
| SigLIP 模型 | `/data/models/siglip-base-patch16-384/` |
| K8S 清单 | `/data/app/k8s/` |
| Docker 构建 | `/data/app/docker/` |
| 微服务代码 | `/data/app/workers/` |

### Docker 镜像备份

| 镜像 | 版本 | 下载日志 | 用途 |
|------|------|----------|------|
| vllm/vllm-openai | v0.6.6.post1 | 已下载 | 旧版本，不支持 Qwen3-VL |
| vllm/vllm-openai | **v0.14.1** | `/tmp/vllm-v0.14.1-pull.log` | **推荐版本，支持 Qwen3-VL** |

### vLLM 启动命令 (T4 兼容)

```bash
# T4 GPU 需要 TRITON_ATTN 后端 + enforce-eager
CUDA_VISIBLE_DEVICES=2,3 \
VLLM_ATTENTION_BACKEND=TRITON_ATTN \
VLLM_USE_FLASHINFER_SAMPLER=0 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name qwen3-vl-32b \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2 \
  --port 8000 \
  --trust-remote-code \
  --enforce-eager
```

### K8S Service 命名规范

**重要**: 避免使用 `vllm` 作为 Service 名称，因为 K8S 会自动生成 `VLLM_SERVICE_HOST` 等环境变量，与 vLLM 自身的 `VLLM_*` 变量冲突。建议使用 `llm-server` 或 `qwen-vl-server`。

---

## RTSP 视频流监控配置

### 摄像头列表 (29路)

RTSP URL 格式: `rtsp://admin:baidu123@{IP}:554/Streaming/Channels/102?transportmode=unicast`

| 路口编号 | IP 地址 | RTSP URL |
|---------|---------|----------|
| 147-01 | 172.21.14.129 | rtsp://admin:baidu123@172.21.14.129:554/Streaming/Channels/102?transportmode=unicast |
| 147-02 | 172.21.14.130 | rtsp://admin:baidu123@172.21.14.130:554/Streaming/Channels/102?transportmode=unicast |
| 147-03 | 172.21.14.131 | rtsp://admin:baidu123@172.21.14.131:554/Streaming/Channels/102?transportmode=unicast |
| 147-04 | 172.21.14.132 | rtsp://admin:baidu123@172.21.14.132:554/Streaming/Channels/102?transportmode=unicast |
| 147-05 | 172.21.14.133 | rtsp://admin:baidu123@172.21.14.133:554/Streaming/Channels/102?transportmode=unicast |
| 147-06 | 172.21.14.134 | rtsp://admin:baidu123@172.21.14.134:554/Streaming/Channels/102?transportmode=unicast |
| 147-07 | 172.21.14.136 | rtsp://admin:baidu123@172.21.14.136:554/Streaming/Channels/102?transportmode=unicast |
| 147-08 | 172.21.14.137 | rtsp://admin:baidu123@172.21.14.137:554/Streaming/Channels/102?transportmode=unicast |
| 147-09 | 172.21.14.139 | rtsp://admin:baidu123@172.21.14.139:554/Streaming/Channels/102?transportmode=unicast |
| 147-10 | 172.21.14.140 | rtsp://admin:baidu123@172.21.14.140:554/Streaming/Channels/102?transportmode=unicast |
| 147-11 | 172.21.14.141 | rtsp://admin:baidu123@172.21.14.141:554/Streaming/Channels/102?transportmode=unicast |
| 146-01 | 172.21.15.1 | rtsp://admin:baidu123@172.21.15.1:554/Streaming/Channels/102?transportmode=unicast |
| 146-02 | 172.21.15.2 | rtsp://admin:baidu123@172.21.15.2:554/Streaming/Channels/102?transportmode=unicast |
| 146-03 | 172.21.15.3 | rtsp://admin:baidu123@172.21.15.3:554/Streaming/Channels/102?transportmode=unicast |
| 146-04 | 172.21.15.4 | rtsp://admin:baidu123@172.21.15.4:554/Streaming/Channels/102?transportmode=unicast |
| 146-05 | 172.21.15.5 | rtsp://admin:baidu123@172.21.15.5:554/Streaming/Channels/102?transportmode=unicast |
| 146-06 | 172.21.15.7 | rtsp://admin:baidu123@172.21.15.7:554/Streaming/Channels/102?transportmode=unicast |
| 146-07 | 172.21.15.8 | rtsp://admin:baidu123@172.21.15.8:554/Streaming/Channels/102?transportmode=unicast |
| 146-08 | 172.21.15.9 | rtsp://admin:baidu123@172.21.15.9:554/Streaming/Channels/102?transportmode=unicast |
| 146-09 | 172.21.15.12 | rtsp://admin:baidu123@172.21.15.12:554/Streaming/Channels/102?transportmode=unicast |
| 146-10 | 172.21.15.13 | rtsp://admin:baidu123@172.21.15.13:554/Streaming/Channels/102?transportmode=unicast |
| 146-11 | 172.21.15.14 | rtsp://admin:baidu123@172.21.15.14:554/Streaming/Channels/102?transportmode=unicast |
| 146-12 | 172.21.15.15 | rtsp://admin:baidu123@172.21.15.15:554/Streaming/Channels/102?transportmode=unicast |
| 146-13 | 172.21.15.16 | rtsp://admin:baidu123@172.21.15.16:554/Streaming/Channels/102?transportmode=unicast |
| 146-14 | 172.21.15.19 | rtsp://admin:baidu123@172.21.15.19:554/Streaming/Channels/102?transportmode=unicast |
| 146-15 | 172.21.15.20 | rtsp://admin:baidu123@172.21.15.20:554/Streaming/Channels/102?transportmode=unicast |
| 146-16 | 172.21.15.21 | rtsp://admin:baidu123@172.21.15.21:554/Streaming/Channels/102?transportmode=unicast |
| 146-17 | 172.21.15.22 | rtsp://admin:baidu123@172.21.15.22:554/Streaming/Channels/102?transportmode=unicast |
| 146-18 | 172.21.15.23 | rtsp://admin:baidu123@172.21.15.23:554/Streaming/Channels/102?transportmode=unicast |

### 监控参数

| 参数 | 值 |
|------|-----|
| 用户名 | admin |
| 密码 | baidu123 |
| 端口 | 554 |
| 通道 | Channels/102 |
| 传输模式 | unicast |

### 汇总报告配置

| 项目 | 设置 |
|------|------|
| 报告周期 | 每 4 小时 |
| 报告内容 | 事故检测统计、违法行为统计、系统健康状态 |
| 存储位置 | `/data/app/outputs/reports/` |

---

## 待办：磁盘自动清理机制（紧急）

### 背景问题

22 路摄像头 7×24 运行，`/data1` 已用 958GB/1.8TB (58%)。现有清理机制有 3 个盲区，不处理会在 1-2 天内写满磁盘。

### 现有清理机制与盲区

| 数据路径 | 当前清理者 | 状态 |
|----------|-----------|------|
| `/data1/videos/rtsp_recordings/seg_*.mp4` | rtsp_ingest（时间戳 > 2h） | ✅ OK |
| `/data1/videos/windows/window_*.mp4` | **无** | **危险，无清理** |
| `/data1/frames/{job_id}/` | semantic_base（分析完成后） | 有 orphan 风险 |
| `/data1/results/` | **无** | 累积中 (217MB) |
| `/data1/positive_events/` | **无** | 累积中 (4.2GB) |

### 方案：K8S CronJob + 独立清理脚本

**设计原则**：只添加新文件，不修改现有 worker 代码。

#### 保留策略

| 数据类型 | 保留时间 | 理由 |
|----------|----------|------|
| RTSP 分片 (seg_*.mp4) | 2h | 已有实现，保持不变 |
| Window 文件 (window_*.mp4) | 2h | 与 RTSP 分片对齐 |
| 帧目录 (/data1/frames/*) | 30min orphan 清理 | 正常处理 < 5min，30min 足够 Event Collector 复制 |
| 分析结果 (/data1/results/) | 7 天 | 调试和审计需要 |
| 阳性事件 (/data1/positive_events/) | 30 天 | 事故证据保留期 |

#### 紧急磁盘压力保护

磁盘剩余 < 100GB 时启用激进模式：window 保留 30min、orphan 帧 10min、results 3 天。

#### 实现步骤

1. **新建 `workers/storage_cleanup.py`**：独立清理脚本，基于 mtime 判断文件年龄，支持 `--dry-run`，flock 防并发，JSON 日志输出
2. **新建 `k8s/30-storage-cleanup.yaml`**：CronJob 每 30 分钟执行，`concurrencyPolicy: Forbid`，`activeDeadlineSeconds: 600`
3. **部署验证**：`kubectl create job --from=cronjob/storage-cleanup test-cleanup -n traffic-vlm`

#### 涉及文件

| 文件 | 操作 |
|------|------|
| `workers/storage_cleanup.py` | 新建 |
| `k8s/30-storage-cleanup.yaml` | 新建 |
| `docker/analyzer-v2.Dockerfile` | 可能修改（确保 COPY 进镜像） |

<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **qwen235b** (41082 symbols, 45369 relationships, 300 execution flows).

GitNexus provides a knowledge graph over this codebase — call chains, blast radius, execution flows, and semantic search.

## Always Start Here

For any task involving code understanding, debugging, impact analysis, or refactoring, you must:

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/refactoring/SKILL.md` |

## Tools Reference

| Tool | What it gives you |
|------|-------------------|
| `query` | Process-grouped code intelligence — execution flows related to a concept |
| `context` | 360-degree symbol view — categorized refs, processes it participates in |
| `impact` | Symbol blast radius — what breaks at depth 1/2/3 with confidence |
| `detect_changes` | Git-diff impact — what do your current changes affect |
| `rename` | Multi-file coordinated rename with confidence-tagged edits |
| `cypher` | Raw graph queries (read `gitnexus://repo/{name}/schema` first) |
| `list_repos` | Discover indexed repos |

## Resources Reference

Lightweight reads (~100-500 tokens) for navigation:

| Resource | Content |
|----------|---------|
| `gitnexus://repo/{name}/context` | Stats, staleness check |
| `gitnexus://repo/{name}/clusters` | All functional areas with cohesion scores |
| `gitnexus://repo/{name}/cluster/{clusterName}` | Area members |
| `gitnexus://repo/{name}/processes` | All execution flows |
| `gitnexus://repo/{name}/process/{processName}` | Step-by-step trace |
| `gitnexus://repo/{name}/schema` | Graph schema for Cypher |

## Graph Schema

**Nodes:** File, Function, Class, Interface, Method, Community, Process
**Edges (via CodeRelation.type):** CALLS, IMPORTS, EXTENDS, IMPLEMENTS, DEFINES, MEMBER_OF, STEP_IN_PROCESS

```cypher
MATCH (caller)-[:CodeRelation {type: 'CALLS'}]->(f:Function {name: "myFunc"})
RETURN caller.name, caller.filePath
```

<!-- gitnexus:end -->
