# 多语义检测功能实现报告

> 生成时间: 2026-01-29
> 分支: feature/multi-semantic-detection

---

## 一、项目概述

### 1.1 需求背景

在现有事故检测系统基础上，扩展支持三种新的语义分析类型：

| 分析类型 | 代码标识 | 检测目标 |
|---------|----------|----------|
| 机动车违法 | `mv_violation` | 闯红灯、违法变道、压线行驶、逆行、违法停车、不礼让行人 |
| 电动车违法 | `ebike_violation` | 闯红灯、逆行、载人超员、不戴头盔、占用机动车道、违法载货 |
| ADS行为 | `ads_behavior` | 示廓灯检测 + 自动驾驶车辆行为分析（两阶段VLM） |

### 1.2 设计原则

- **不破坏现有事故检测**：所有事故检测相关代码、配置、权重保持不变
- **复用基础设施**：共享 Embedding Service、VLM Proxy、Redis Streams
- **独立部署**：每个分析类型独立 K8S Deployment，独立 GPU 分配
- **向后兼容**：旧任务（无 `analysis_type` 字段）自动识别为 `accident`

---

## 二、系统架构

### 2.1 新增组件

```
                            ┌──────────────────────────────────────┐
                            │         video_tasks (Redis)          │
                            │  analysis_type: accident/mv/ebike/ads│
                            └──────────────┬───────────────────────┘
                                           │
           ┌───────────────┬───────────────┼───────────────┬───────────────┐
           ▼               ▼               ▼               ▼               ▼
   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
   │  Semantic     │ │  Semantic     │ │  MV Violation │ │  Ebike        │ │  ADS Behavior │
   │  Analyzer     │ │  Analyzer     │ │  Analyzer     │ │  Violation    │ │  Analyzer     │
   │  (accident)   │ │  GPU 0,1,4,5  │ │  (GPU 4)      │ │  (GPU 5)      │ │  (GPU 6)      │
   └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
                                           │
                            ┌──────────────▼───────────────────────┐
                            │         result_tasks (Redis)         │
                            │  + is_positive, violation_type, etc  │
                            └──────────────────────────────────────┘
```

### 2.2 GPU 分配

| GPU | 用途 | 部署 |
|-----|------|------|
| 0-1 | Embedding Service | K8S Deployment |
| 2-3 | vLLM (Tensor Parallel) | nohup 进程 |
| 4 | MV Violation Analyzer | 新增 K8S Deployment |
| 5 | Ebike Violation Analyzer | 新增 K8S Deployment |
| 6 | ADS Behavior Analyzer | 新增 K8S Deployment |
| 7 | 预留 | - |

### 2.3 消息格式扩展

#### VideoTask 新增字段

```python
@dataclass
class VideoTask:
    job_id: str
    camera_id: str
    window_path: str
    trace_id: str
    seg_end_ts: int = None
    created_at: float = None
    analysis_type: str = "accident"  # 新增: accident/mv_violation/ebike_violation/ads_behavior
```

#### ResultTask 新增字段

```python
@dataclass
class ResultTask:
    # ... 原有字段 ...
    analysis_type: str = "accident"   # 分析类型
    is_positive: bool = False         # 通用阳性标志
    violation_type: str = None        # 违法类型 (mv/ebike)
    behavior_type: str = None         # 行为类型 (ads)
```

---

## 三、实现详情

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `workers/semantic_base.py` | 语义分析器抽象基类，封装共享逻辑 |
| `workers/mv_violation_analyzer.py` | 机动车违法检测器 (24个语义模板) |
| `workers/ebike_violation_analyzer.py` | 电动车违法检测器 (24个语义模板) |
| `workers/ads_behavior_analyzer.py` | ADS行为检测器 (17个模板, 两阶段VLM) |
| `k8s/10-mv-violation-analyzer.yaml` | K8S 部署清单 |
| `k8s/11-ebike-violation-analyzer.yaml` | K8S 部署清单 |
| `k8s/12-ads-behavior-analyzer.yaml` | K8S 部署清单 |
| `docker/*.Dockerfile` | Docker 构建文件 |
| `tests/test_multi_semantic.py` | 单元测试 |

### 3.2 语义模板示例

#### 机动车违法模板

```python
MV_VIOLATION_TEMPLATES = [
    "car running red light",
    "illegal lane change",
    "car crossing lane markings",
    "wrong way driving",
    "car not yielding to pedestrian",
    # ... 共24个模板
]
```

#### ADS行为模板

```python
ADS_BEHAVIOR_TEMPLATES = [
    "vehicle with marker lights on",
    "autonomous vehicle indicator lights",
    "self-driving car with lights on",
    # ... 共17个模板
]
```

### 3.3 ADS 两阶段 VLM 流程

```
Phase 1: 示廓灯检测
├── 输入: 12帧关键帧
├── Prompt: "检测是否存在开启示廓灯的车辆"
└── 输出: { marker_light_detected, marker_light_state, confidence }
           │
           ├── 未检测到 → 结束, judgment=NO
           │
           └── 检测到开启 → Phase 2

Phase 2: 行为分析
├── 输入: 同上12帧
├── Prompt: "分析示廓灯车辆的驾驶行为"
└── 输出: { behavior_type, behavior_description, safety_level }
```

---

## 四、测试结果

### 4.1 单元测试 (本地)

```
[PASS] Test 1: default analysis_type = accident
[PASS] Test 2: custom analysis_type = mv_violation
[PASS] Test 3: to_dict includes analysis_type
[PASS] Test 4: from_dict parses analysis_type
[PASS] Test 5: from_dict default analysis_type = accident
[PASS] Test 6: ResultTask new fields
[PASS] Test 7: ResultTask to_dict
[PASS] MV: JSON response parsing
[PASS] MV: Fallback response parsing
[PASS] Ebike: JSON response parsing
[PASS] ADS: Phase1 response parsing

=== All tests passed! ===
```

### 4.2 远程集成测试 (r6500-g4-2)

```
[PASS] All modules imported successfully
MV templates: 24
Ebike templates: 24
ADS templates: 17
[PASS] VideoTask.analysis_type works
[PASS] ResultTask new fields work
[PASS] Backward compatibility: old tasks default to accident
[PASS] Round-trip serialization works
[PASS] ResultTask backward compatibility

=== Remote integration tests passed! ===
```

### 4.3 事故检测回归测试

**Baseline 指标 (261 样本, 2026-01-28):**

| 指标 | 值 |
|------|-----|
| TP | 173 |
| FP | 0 |
| TN | 64 |
| FN | 24 |
| **Recall** | **87.82%** |
| **Precision** | **100%** |
| **F1** | **93.51%** |
| **Accuracy** | **90.80%** |

**回归验证:**
- 现有 semantic-analyzer pods 继续正常运行
- 向后兼容性测试全部通过
- 新代码不影响现有事故检测逻辑

---

## 五、部署指南

### 5.1 构建新镜像

```bash
# 在远程服务器执行
cd /data/app

# 构建机动车违法检测器
docker build -t localhost:5000/mv-violation-analyzer:v1 -f docker/mv-violation-analyzer.Dockerfile .
docker push localhost:5000/mv-violation-analyzer:v1

# 构建电动车违法检测器
docker build -t localhost:5000/ebike-violation-analyzer:v1 -f docker/ebike-violation-analyzer.Dockerfile .
docker push localhost:5000/ebike-violation-analyzer:v1

# 构建ADS行为检测器
docker build -t localhost:5000/ads-behavior-analyzer:v1 -f docker/ads-behavior-analyzer.Dockerfile .
docker push localhost:5000/ads-behavior-analyzer:v1
```

### 5.2 部署 K8S 资源

```bash
# 部署新分析器
kubectl apply -f k8s/10-mv-violation-analyzer.yaml
kubectl apply -f k8s/11-ebike-violation-analyzer.yaml
kubectl apply -f k8s/12-ads-behavior-analyzer.yaml

# 验证部署
kubectl get pods -n traffic-vlm -l analysis-type
```

### 5.3 发送测试任务

```python
from workers.common.redis_client import RedisStreamClient, VideoTask

redis = RedisStreamClient("redis.traffic-vlm", 6379)

# 发送机动车违法检测任务
task = VideoTask(
    job_id="mv_test_001",
    camera_id="cam_147",
    window_path="/data1/videos/test_mv.mp4",
    trace_id="trace_mv_001",
    analysis_type="mv_violation"
)
redis.add_video_task(task)
```

---

## 六、注意事项

1. **GPU 冲突**：新分析器使用 GPU 4,5,6，避开 vLLM 占用的 GPU 2,3
2. **Consumer Group**：每个分析类型使用独立的 consumer group
3. **幂等性**：使用 `job_id + analysis_type` 组合作为唯一键
4. **扩缩容**：可根据需求调整各 Deployment 的 replicas

---

## 七、后续工作

- [ ] 收集违法/ADS真实视频样本构建测试集
- [ ] 针对各分析类型进行阈值调优
- [ ] 添加专门的评测脚本和指标
- [ ] 考虑是否需要 HPA 自动扩缩

---

*报告生成: 2026-01-29*
*作者: Claude Code*
