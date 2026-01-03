\#目标：我想要调用Qwen3-VL-235B-A22B-Instruct这个模型，我想上传一个路测的视频，然后让Qwen3-VL-235B-A22B-Instruct这个模型对视频进行理解和分析。



\#方法：

&nbsp;##1 参考官网提供的调用方法，代码和说明网址是：https://modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct。

\##2 你去网络上深度搜索成功调用的方法。



\#结果：

&nbsp;最终生成一个可以跟Qwen3-VL-235B-A22B-Instruct对话的网页类似chatbox，让我可以上传视频，并拿到返回的结果。



\#重要规则：

\##虚拟环境使用规范

**强制要求**：本项目所有代码执行和依赖安装操作，必须在虚拟环境中进行。

- **测试代码**：必须使用 `d:/project2025/qwen235b/venv/Scripts/python.exe`
- **正式运行**：必须使用 `d:/project2025/qwen235b/venv/Scripts/python.exe`
- **安装依赖**：必须使用 `d:/project2025/qwen235b/venv/Scripts/pip.exe`

**原因**：
1. 系统Python环境缺少CUDA支持的PyTorch，会导致GPU加速失效
2. 虚拟环境包含完整的项目依赖（torch+cu121、transformers等）
3. 使用错误的Python会导致性能下降30-50倍

**禁止**：
- ❌ 直接使用 `python` 命令（可能调用系统Python）
- ❌ 使用系统Python路径 `C:\Users\ALIENWARE\AppData\Local\Programs\Python\Python310\python.exe`

**正确示例**：
```bash
# 运行评测
d:/project2025/qwen235b/venv/Scripts/python.exe run_full_eval.py --dump-video-results

# 安装依赖
d:/project2025/qwen235b/venv/Scripts/pip.exe install package-name

# 运行测试
d:/project2025/qwen235b/venv/Scripts/python.exe test_xxx.py
```

## 日志输出长度限制

**强制要求**：在测试或执行代码时，监控的日志输出不要超过窗口上下文的最大长度。

**具体措施**：
- 长时间运行的任务使用后台模式（`run_in_background`），通过 `TaskOutput` 分批获取结果
- 避免一次性输出大量日志，必要时使用 `block=false` 非阻塞检查状态
- 对于批量处理任务，优先查看最终统计结果而非逐条日志
- 如果日志过长导致截断，应主动查询结果文件或数据库获取完整信息

## 评测输出规范

**强制要求**：**每次回归测试**以及**任何分析**都必须生成以下追踪文件：
1. **Casebook (Markdown)** - FN/FP详细案例分析
2. **Decision Trace (JSON)** - 决策链追踪

这是为了便于归因分析、复现问题和持续改进。**禁止跳过这些输出。**

### 必需输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `summary.json` | JSON | 汇总指标 (TP/FP/TN/FN, Recall, FPR等) |
| `per_file.json` | JSON | 每个视频的预测结果和decision_reason |
| `casebook.md` | Markdown | FN/FP案例详细分析 (含VLM响应、关键帧路径) |
| `decision_trace.json` | JSON | 每个视频的决策追踪链 |
| `video_results/*.result.json.gz` | GZIP JSON | 完整pipeline输出 (需 `--dump-video-results`) |

### Casebook 格式规范

```markdown
# Casebook - FN/FP 案例分析

## FN-1: 101.mp4
- **路径**: `uploads/事故数据集/101.mp4`
- **Ground Truth**: 事故
- **Prediction**: NO
- **Decision Reason**: 无clip通过阈值

### Clip 分析
| clip_id | base_score | final_score | VLM verdict | confidence |
|---------|------------|-------------|-------------|------------|
| clip-xxx | 0.64 | 0.77 | NO | 0.95 |

### VLM 响应
- **S1**: verdict=NO, confidence=0.95
- **S2**: verdict=UNCERTAIN, confidence=0.65
- **Final**: NO
- **Summary**: "监控画面显示夜间雨天道路..."

### 关键帧
- `data/camera-1/20251231/annotated_frames/clip-xxx/frame_00.jpg`
- `data/camera-1/20251231/annotated_frames/clip-xxx/frame_05.jpg`
```

### Decision Trace 格式规范

```json
{
  "video_name": "101.mp4",
  "ground_truth": true,
  "predicted": false,
  "decision_chain": [
    {"stage": "clip_generation", "clips_count": 1, "scores": [0.64]},
    {"stage": "coverage_scoring", "final_scores": [0.77], "threshold": 0.35},
    {"stage": "threshold_filter", "passed": 1, "skipped": 0},
    {"stage": "vlm_s1", "verdict": "NO", "confidence": 0.95},
    {"stage": "escalation", "triggered": true, "reason": "risk=1.00>=0.6"},
    {"stage": "vlm_s2", "verdict": "UNCERTAIN", "confidence": 0.65},
    {"stage": "final_decision", "verdict": "NO", "logic": "S1=NO takes precedence"}
  ]
}
```

### 评测命令

```bash
# 完整评测 (含casebook和decision_trace)
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/run_xxx \
  --dump-video-results

# 启用Top-1 Fallback
d:/project2025/qwen235b/venv/Scripts/python.exe tools/run_eval_to_output.py \
  --output-dir outputs/run_xxx \
  --dump-video-results \
  --enable-fallback
```

