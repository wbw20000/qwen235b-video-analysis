# 评分与决策规则清单

> 自动生成时间: 2025-12-26
> 来源代码: traffic_vlm/file_event_locator.py, traffic_vlm/coverage_scorer.py, traffic_vlm/config.py

---

## 1. 候选生成规则

### 1.1 输入来源
- 单个 mp4 文件作为独立样本
- 目录: `data/camera-<id>/<date>/raw_suspect_clips/`

### 1.2 候选模式
配置项: `FileEventLocatorConfig.candidate_mode`

| 模式 | 说明 |
|------|------|
| `file_only` | 仅生成文件级候选（整个mp4作为一个候选） |
| `file_plus_subclips` | 文件级候选 + 围绕风险峰值的子候选窗口 |

### 1.3 子候选窗口参数
配置来源: `FileEventLocatorConfig` (config.py:188-212)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pre_roll` | 8.0s | 事故前覆盖时长 |
| `post_roll` | 12.0s | 事故后覆盖时长 |
| `merge_gap_sec` | 3.0s | 子候选合并间隔 |
| `top_n_peaks` | 5 | TopN风险峰值数量 |
| `peak_min_distance_sec` | 3.0s | 峰值间最小间距 |
| `peak_threshold` | 0.15 | 峰值阈值(0~1) |
| `risk_sampling_fps` | 2.0 | 风险时间序列采样频率(Hz) |

---

## 2. 过滤规则（Gate）

### 2.1 clip_score 阈值过滤
配置来源: `VLMConfig` (config.py:71-96)

```
if skip_low_score_vlm AND clip_score < clip_score_threshold:
    SKIP clip, reason = "clip_score={clip_score} < {threshold}"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `clip_score_threshold` | 0.35 | clip_score低于此值跳过VLM |
| `skip_low_score_vlm` | True | 是否启用阈值过滤 |

应用阶段: pipeline.py 预处理阶段（阶段A）

### 2.2 Top-K 选择
配置来源: `VLMConfig.top_clips` (config.py:76)

```
clips_to_process = suspect_clips[:top_clips]  # 默认 top_clips = 3
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_clips` | 3 | 只预处理排名前N的clips |

### 2.3 进入VLM的必要条件
1. clip 在 Top-K 范围内
2. `clip_score >= clip_score_threshold` (若启用)
3. 预处理成功（抽帧、标注完成）

---

## 3. 打分公式

### 3.1 base_score 定义
来源: `file_event_locator.py:603`

```python
base_score = risk_result.max_risk  # 风险时间序列的最大值
```

其中 `max_risk` 来自 `compute_risk_timeline()` 函数，基于YOLO检测的目标距离/速度计算。

### 3.2 coverage 相关分数

#### 3.2.1 coverage_effective 计算
来源: `file_event_locator.py:588-592`

```python
# t0: 碰撞时刻估计（秒）
# duration: 视频时长（秒）
# pre_roll, post_roll: 目标覆盖窗口（默认8s, 12s）

pre_ok = min(1.0, t0 / pre_roll)           # t0前有多少覆盖（0~1）
post_ok = min(1.0, (duration - t0) / post_roll)  # t0后有多少覆盖（0~1）
coverage_raw = pre_ok * post_ok            # 原始覆盖分数（0~1）
coverage_effective = t0_validity * coverage_raw  # 有效覆盖分数
```

示例:
- t0=3s, duration=64s, pre_roll=8s, post_roll=12s
- pre_ok = min(1.0, 3/8) = 0.375
- post_ok = min(1.0, 61/12) = 1.0
- coverage_raw = 0.375 * 1.0 = 0.375
- 若 t0_validity=0.39, coverage_effective = 0.39 * 0.375 = 0.146

#### 3.2.2 t0_validity 计算
来源: `file_event_locator.py:300-400`, `compute_t0_validity()`

```python
# 各项证据权重
validity = (
    0.25 * min_distance_evidence +    # 最小距离证据
    0.30 * distance_drop_evidence +   # 距离突降证据
    0.25 * velocity_change_evidence + # 速度变化证据
    0.10 * iou_contact_evidence +     # IoU接触证据
    0.10 * tracking_jitter_evidence   # 跟踪抖动证据
)
```

证据阈值:
| 证据 | 阈值参数 | 默认值 |
|------|----------|--------|
| min_distance | `min_distance_threshold` | 150px |
| distance_drop | `distance_drop_threshold` | 0.3 |
| velocity_change | `velocity_change_threshold` | 0.4 |
| iou_contact | `iou_contact_threshold` | 0.05 |

### 3.3 late_start_penalty 计算
来源: `file_event_locator.py:599`

```python
late_start_penalty = max(0, pre_roll - t0)
```

示例:
- pre_roll=8s, t0=3s → penalty = max(0, 8-3) = 5
- pre_roll=8s, t0=10s → penalty = max(0, 8-10) = 0

### 3.4 final_score 计算

#### 3.4.1 file_event_locator 中的公式
来源: `file_event_locator.py:605`

```python
final_score = base_score + 0.3 * coverage_effective - 0.02 * late_start_penalty
```

| 权重 | 值 | 说明 |
|------|-----|------|
| w_base | 1.0 | base_score系数 |
| w_coverage | 0.3 | coverage_effective系数 |
| w_late | 0.02 | late_start_penalty惩罚系数 |

#### 3.4.2 coverage_scorer 中的公式（备选）
来源: `coverage_scorer.py:286-289`

```python
final_score = base_score + lambda_coverage * coverage_score - mu_late * late_start_penalty
```

| 权重 | 配置项 | 默认值 |
|------|--------|--------|
| lambda_coverage | `CoverageConfig.lambda_coverage` | 0.20 |
| mu_late | `CoverageConfig.mu_late` | 0.03 |

### 3.5 is_full_process 判定
来源: `file_event_locator.py:611-615`

```python
is_full_process = (pre_ok >= 0.8 AND post_ok >= 0.8 AND t0_validity >= 0.3)
```

---

## 4. 排序规则（语义对齐版 v2）

> **重大更新**: 使用单一 `rank_score` 排序，禁止 tuple sort_key

### 4.1 rank_score 计算（唯一排序主分）
来源: `file_event_locator.py:791-926`

```python
rank_score = final_score + full_process_bonus - post_event_penalty + verdict_bonus
```

| 组成部分 | 公式 | 说明 |
|----------|------|------|
| `final_score` | base + 0.3*cover_eff - 0.02*late_penalty | 基础最终分 |
| `full_process_bonus` | B * full_process_score | 完整覆盖加分 (B=0.20) |
| `post_event_penalty` | P * post_event_score | 仅后果惩罚 (P=0.20) |
| `verdict_bonus` | 根据verdict | YES=0.10, UNCERTAIN=0.05, 其他=0 |

### 4.2 full_process_score 计算（三因子乘积）
```python
# 仅当 verdict = YES 时计算，否则为 0
full_process_score = pre_score * impact_score * post_score

pre_score = min(1.0, t0 / pre_roll)           # 碰撞前覆盖 (pre_roll=8s)
impact_score = t0_validity                     # 碰撞瞬间证据强度
post_score = min(1.0, (duration - t0) / post_roll)  # 碰撞后覆盖 (post_roll=12s)
```

### 4.3 post_event_score 定义
```python
post_event_score = 1.0 if verdict == "POST_EVENT_ONLY" else 0.0
```

### 4.4 语义约束表

| verdict | full_process_score | post_event_score | verdict_bonus | 说明 |
|---------|-------------------|------------------|---------------|------|
| `YES` | pre*imp*post | 0 | 0.10 | 完整覆盖事故过程 |
| `POST_EVENT_ONLY` | 0 | 1 | 0.05 | 仅捕捉事故后果 |
| `UNCERTAIN` | 0 | 0 | 0.05 | 需人工复核 |
| `NO` | 0 | 0 | 0 | 无事故 |

### 4.5 权重参数
配置来源: `RankScoreConfig` (config.py:215-243)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `B_full_process` | 0.20 | full_process_bonus权重 |
| `P_post_event` | 0.20 | post_event_penalty权重 |
| `confirm_verdict_bonus` | 0.10 | verdict=YES加分 |
| `uncertain_verdict_bonus` | 0.05 | verdict=UNCERTAIN加分 |
| `pre_roll` | 8.0s | 碰撞前目标覆盖时长 |
| `post_roll` | 12.0s | 碰撞后目标覆盖时长 |

### 4.6 排序代码（禁止tuple sort_key）
```python
# 单一字段排序
sorted(candidates, key=lambda c: c.get("rank_score", 0.0), reverse=True)
```

---

## 5. VLM 规则

### 5.1 verdict schema (四态)
来源: `vlm_client.py`

| verdict | 说明 |
|---------|------|
| `YES` | 确认事故 - 看到碰撞过程 |
| `POST_EVENT_ONLY` | 仅后果 - 未看到碰撞但有明确后果 |
| `UNCERTAIN` | 不确定，需人工复核 |
| `NO` | 确认无事故 |

### 5.2 VLM 对 kept 的影响
- `verdict ∈ {YES, POST_EVENT_ONLY, UNCERTAIN}` → kept=True
- `verdict = NO` → 有条件保留（见下节）

---

## 6. 保留策略（kept）

### 6.1 配置来源
`VLMRetentionConfig` (config.py:165-185)

### 6.2 保留规则

```python
# verdict ∈ {YES, POST_EVENT_ONLY, UNCERTAIN}
if verdict in ["YES", "POST_EVENT_ONLY", "UNCERTAIN"]:
    kept = True
    keep_reason = f"verdict={verdict}_confirmed"

# verdict = NO 时的条件保留
elif verdict == "NO":
    if t0_validity >= validity_threshold:
        kept = True
        keep_reason = f"NO_but_kept: validity={t0_validity}>=0.3"
    elif risk_peak >= risk_peak_threshold:
        kept = True
        keep_reason = f"NO_but_kept: risk_peak={risk_peak}>=0.25"
    elif roi_median >= roi_median_threshold:
        kept = True
        keep_reason = f"NO_but_kept: roi_median={roi_median}>=80"
    else:
        kept = False
```

### 6.3 保留阈值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `validity_threshold` | 0.3 | t0_validity保留阈值 |
| `risk_peak_threshold` | 0.25 | risk_peak保留阈值 |
| `roi_median_threshold` | 80.0 | ROI中值边长阈值(像素) |
| `accident_score_threshold` | 1.0 | accident_score保留阈值 |
| `force_keep_on_uncertain` | True | UNCERTAIN强制保留 |
| `force_keep_on_post_event` | True | POST_EVENT_ONLY强制保留 |

---

## 7. 默认参数来源

### 7.1 配置文件层级

1. **代码默认值**: dataclass field default
2. **配置文件**: `config.yaml` (如果存在)
3. **命令行参数**: 覆盖配置文件

### 7.2 关键配置类

| 配置类 | 文件位置 | 说明 |
|--------|----------|------|
| `VLMConfig` | config.py:71-96 | VLM调用参数 |
| `CoverageConfig` | config.py:117-138 | 覆盖度评分参数 |
| `VLMRetentionConfig` | config.py:165-185 | 保留策略参数 |
| `FileEventLocatorConfig` | config.py:188-212 | 文件级事件定位参数 |
| `TrajectoryScoreConfig` | config.py:100-114 | 轨迹碰撞评分参数 |

---

## 附录: 关键函数索引

| 函数 | 文件:行号 | 说明 |
|------|-----------|------|
| `compute_risk_timeline()` | file_event_locator.py:87 | 计算风险时间序列 |
| `compute_t0_validity()` | file_event_locator.py:300 | 验证t0有效性 |
| `generate_file_candidates()` | file_event_locator.py:520 | 生成文件级候选 |
| `rank_candidates()` | file_event_locator.py:791 | 排序候选 |
| `apply_conditional_retention()` | file_event_locator.py:750 | 应用保留策略 |
| `compute_coverage_score()` | coverage_scorer.py:180 | 计算覆盖度分数 |
| `compute_final_score()` | coverage_scorer.py:265 | 计算最终分数 |
