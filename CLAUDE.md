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

## 待办：消融实验（验证物理检测贡献）

### 背景问题

当前 `clip_score` 计算中存在**信号冗余**：
- `similarity_score`：SigLIP 语义相似度（包含事故模板匹配）
- `accident_template_hit`：命中事故模板时直接给 1.0

```python
# temporal_clusterer.py
def _frame_accident_score(frame):
    if meta.get("accident_template_hit"):
        scores.append(1.0)  # 直接给1.0，淹没物理信号
    return max(scores)
```

**问题**：只要命中事故模板，物理检测信号（collision/intersection/deceleration）就被淹没，可能没有实际贡献。

### 消融实验设计

| 实验 | 修改 | 验证目标 |
|------|------|---------|
| **A1: 禁用轨迹评分** | `trajectory_score.enabled = False` | 物理检测是否有用 |
| **A2: 禁用 template_hit** | 注释 `accident_template_hit` 逻辑 | 信号冗余是否影响效果 |
| **A3: 仅物理信号** | 移除 `accident_template_hit`，只用轨迹分数 | 物理检测能否独立工作 |

### 实验命令

```bash
# 基线
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_baseline

# A1: 禁用轨迹评分
# 修改 config.py: trajectory_score.enabled = False
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_no_trajectory

# A2: 禁用 template_hit
# 修改 temporal_clusterer.py: 注释 accident_template_hit 相关代码
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/ablation_no_template_hit
```

### 预期结论

| 实验结果 | 说明 | 后续动作 |
|---------|------|---------|
| A1效果不变 | 物理检测无用 | 可移除省算力（YOLO少跑一次） |
| A1效果下降 | 物理检测对边界案例有帮助 | 保留但优化权重 |
| A2效果更好 | 当前设计有冗余 | 重构 clip_score 计算逻辑 |

### 相关文件

- `traffic_vlm/temporal_clusterer.py`: `_frame_accident_score()` 函数
- `traffic_vlm/trajectory_scorer.py`: 轨迹评分计算
- `traffic_vlm/config.py`: `TrajectoryScoreConfig.enabled`

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
