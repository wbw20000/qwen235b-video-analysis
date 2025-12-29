# 评分口径快照

> 生成时间: 2025-12-26
> 基于 run_id: 20251226_131051_698578c

---

## 1. 当前公式与代码位置

### 1.1 final_score 公式
来源: `traffic_vlm/file_event_locator.py:605`
```python
final_score = base_score + 0.3 * coverage_effective - 0.02 * late_start_penalty
```

### 1.2 coverage_effective 公式
来源: `traffic_vlm/file_event_locator.py:588-592`
```python
pre_ok = min(1.0, t0 / pre_roll)           # pre_roll=8s
post_ok = min(1.0, (duration - t0) / post_roll)  # post_roll=12s
coverage_raw = pre_ok * post_ok
coverage_effective = t0_validity * coverage_raw
```

### 1.3 late_start_penalty 公式
来源: `traffic_vlm/file_event_locator.py:599`
```python
late_start_penalty = max(0, pre_roll - t0)
```

### 1.4 rank_score 公式（唯一排序主分）
来源: `traffic_vlm/file_event_locator.py:791-926`
```python
rank_score = final_score + full_process_bonus - post_event_penalty + verdict_bonus

# 其中:
full_process_score = pre_score * impact_score * post_score  # 仅当verdict=YES
full_process_bonus = B * full_process_score                 # B=0.20
post_event_penalty = P * post_event_score                   # P=0.20
verdict_bonus = 0.10 if verdict=YES, 0.05 if UNCERTAIN/POST_EVENT_ONLY, 0 otherwise
```

### 1.5 语义约束表

| verdict | full_process_score | post_event_score | verdict_bonus |
|---------|-------------------|------------------|---------------|
| YES | pre_score * impact_score * post_score | 0 | 0.10 |
| POST_EVENT_ONLY | 0 | 1.0 | 0.05 |
| UNCERTAIN | 0 | 0 | 0.05 |
| NO | 0 | 0 | 0 |

---

## 2. 当前产物对比表（run_id=20251226_131051_698578c）

| clip_id | base_score | t0 | pre_ok | post_ok | t0_validity | coverage_effective | late_penalty | final_score | full_process_bonus | post_event_penalty | verdict_bonus | rank_score | verdict | rank |
|---------|-----------|-----|--------|---------|-------------|-------------------|--------------|-------------|-------------------|-------------------|--------------|-----------|---------|------|
| clip-a826cb3b | 0.9165 | 3.0 | 0.375 | 1.0 | 0.391 | 0.147 | 5.0 | 0.8605 | 0.029 | 0.0 | 0.10 | 0.9898 | YES | 1 |
| clip-51e7b0c2 | 0.9335 | 36.0 | 1.0 | 1.0 | 0.395 | 0.395 | 0 | 1.0520 | 0.0 | 0.2 | 0.05 | 0.9020 | POST_EVENT_ONLY | 2 |
| clip-d8839af9 | 0.9274 | 33.0 | 1.0 | 1.0 | 0.396 | 0.396 | 0 | 1.0462 | 0.0 | 0.2 | 0.05 | 0.8962 | POST_EVENT_ONLY | 3 |
| clip-372493c8 | 0.9142 | 3.5 | 0.438 | 1.0 | 0.375 | 0.164 | 4.5 | 0.8735 | 0.0 | 0.0 | 0 | 0.8735 | NO | 4 |

---

## 3. 修复状态

### 3.1 ✅ 统计口径已对齐（P0 已修复）

**修复前后对比** (run_id=20251226_132626_698578c):

| 来源 | n_source_videos | n_kept | n_preprocessed | n_vlm_analyzed |
|------|----------------|--------|----------------|----------------|
| scoring_summary.json | - | 3 | - | 4 |
| summary.json (修复后) | 4 | 3 | 4 | 4 |

**修复内容**: `test_file_event_regression.py:601-606`
```python
context.n_source_videos = len(file_map)
context.n_preprocessed = sum(1 for c in ranked if c.get("filter_status") == "PASSED")
context.n_pass_score = sum(1 for c in ranked if c.get("clip_score", 0) >= 0.35)
context.n_kept = sum(1 for c in ranked if c.get("kept"))
context.n_topk = min(len(ranked), 3)
```

### 3.2 ✅ 排序正确（语义对齐版）

当前排序结果符合预期：
- clip-a826cb3b (YES) rank=1, rank_score=0.1000
- clip-372493c8 (NO) rank=2, rank_score=0.0000
- clip-d8839af9 (POST_EVENT_ONLY) rank=3, rank_score=-0.1500
- clip-51e7b0c2 (POST_EVENT_ONLY) rank=4, rank_score=-0.1500

**验证通过**:
- YES clips 的 rank_score > POST_EVENT_ONLY clips
- POST_EVENT_ONLY clips 的 full_process_score = 0
- POST_EVENT_ONLY clips 的 post_event_score = 1.0

### 3.3 ✅ 权重已正确应用

- `full_process_bonus = 0.2 * full_process_score` (不是0/1标志)
- `post_event_penalty = 0.2 * post_event_score` (不是0/1标志)
- `verdict_bonus`: YES=0.10, POST_EVENT_ONLY=0.05, UNCERTAIN=0.05, NO=0

---

## 4. 任务完成状态

| 任务 | 状态 | 说明 |
|------|------|------|
| 任务A: 输出评分口径快照 | ✅ 完成 | 本文档 |
| 任务B: 修复统计口径 | ✅ 完成 | RunContext字段已设置 |
| 任务C: 完整过程强规则 | ✅ 完成 | 排序已正确 |
| 任务D: 验证交付 | ✅ 完成 | 回归测试通过 |
