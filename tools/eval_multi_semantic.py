#!/usr/bin/env python3
"""
多语义检测评测脚本

功能:
- 批量处理测试视频
- 计算 Recall/Precision/F1
- 支持 mv_violation, ebike_violation, ads_behavior

使用方式:
python tools/eval_multi_semantic.py \
    --analysis-type mv_violation \
    --video-dir /data1/testdata/multi_semantic/mv_violation \
    --labels-file /data1/testdata/multi_semantic/mv_violation/labels.json \
    --output-dir outputs/multi_semantic_eval
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@dataclass
class EvalResult:
    """单个视频评测结果"""
    video_path: str
    video_name: str
    ground_truth: bool  # True=阳性, False=阴性
    predicted: bool
    pred_label: str  # YES/NO/UNCERTAIN
    confidence: float
    correct: bool
    violation_type: Optional[str] = None
    reason: str = ""
    processing_time: float = 0.0
    clip_score: float = 0.0
    error: Optional[str] = None


@dataclass
class EvalSummary:
    """评测汇总"""
    analysis_type: str
    total_videos: int
    tp: int = 0  # True Positive
    fp: int = 0  # False Positive
    tn: int = 0  # True Negative
    fn: int = 0  # False Negative
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    fpr: float = 0.0  # False Positive Rate
    processing_time_avg: float = 0.0
    timestamp: str = ""


def evaluate_single_video(
    video_path: str,
    analyzer,
    job_id: str = ""
) -> Dict[str, Any]:
    """
    评测单个视频

    Args:
        video_path: 视频路径
        analyzer: 分析器实例
        job_id: 任务ID

    Returns:
        分析结果字典
    """
    start_time = time.time()
    trace_id = f"eval_{int(time.time()*1000)}"

    try:
        # 1. 抽帧
        frames = analyzer.extract_frames(video_path, fps=1.0, job_id=job_id, trace_id=trace_id)
        if not frames:
            return {
                "judgment": "NO",
                "confidence": 1.0,
                "reason": "抽帧为空",
                "clip_score": 0.0,
                "processing_time": time.time() - start_time,
                "error": "no_frames"
            }

        # 2. 计算嵌入
        frames = analyzer.compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

        # 3. 计算相似度
        frames = analyzer.compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

        # 4. 聚类
        clips = analyzer.cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

        # 5. 获取最高分片段
        best_clip_score = 0.0
        if clips:
            best_clip = max(clips, key=lambda c: c.clip_score)
            best_clip_score = best_clip.clip_score

            # 降低阈值以测试 VLM（原阈值 0.35）
            clip_threshold = 0.10  # 降低阈值以触发更多 VLM 调用
            if best_clip_score >= clip_threshold:
                keyframes = analyzer.select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                best_clip.keyframes = keyframes
                analysis_result = analyzer.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)

                # 清理临时文件
                analyzer._cleanup_frames(frames)

                return {
                    "judgment": analysis_result.judgment,
                    "confidence": analysis_result.confidence,
                    "reason": analysis_result.reason,
                    "violation_type": analysis_result.violation_type,
                    "clip_score": best_clip_score,
                    "processing_time": time.time() - start_time,
                    "error": None
                }

        # 清理临时文件
        analyzer._cleanup_frames(frames)

        return {
            "judgment": "NO",
            "confidence": 1.0,
            "reason": f"最高分 {best_clip_score:.3f} 低于阈值 0.10",
            "clip_score": best_clip_score,
            "processing_time": time.time() - start_time,
            "error": None
        }

    except Exception as e:
        return {
            "judgment": "NO",
            "confidence": 0.0,
            "reason": str(e),
            "clip_score": 0.0,
            "processing_time": time.time() - start_time,
            "error": str(e)
        }


def run_evaluation(
    analysis_type: str,
    video_dir: str,
    labels_file: str,
    output_dir: str,
    embedding_url: str = "http://localhost:8080",
    vlm_proxy_url: str = "http://localhost:8001",
    limit: int = 0
) -> EvalSummary:
    """
    运行评测

    Args:
        analysis_type: 分析类型 (mv_violation, ebike_violation, ads_behavior)
        video_dir: 测试视频目录
        labels_file: 标注文件路径 (JSON格式: {"video_name": true/false})
        output_dir: 输出目录
        embedding_url: Embedding 服务地址
        vlm_proxy_url: VLM 代理地址
        limit: 限制处理视频数量 (0=不限制)

    Returns:
        评测汇总
    """
    # 加载标注
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)

    print(f"[INFO] 加载 {len(labels)} 个视频标注")

    # 创建分析器
    if analysis_type == "mv_violation":
        from workers.mv_violation_analyzer import MVViolationAnalyzer
        analyzer = MVViolationAnalyzer(
            redis_host="localhost",
            redis_port=6379,
            embedding_url=embedding_url,
            vlm_proxy_url=vlm_proxy_url,
            results_dir="/tmp/eval_results"
        )
    elif analysis_type == "ebike_violation":
        from workers.ebike_violation_analyzer import EbikeViolationAnalyzer
        analyzer = EbikeViolationAnalyzer(
            redis_host="localhost",
            redis_port=6379,
            embedding_url=embedding_url,
            vlm_proxy_url=vlm_proxy_url,
            results_dir="/tmp/eval_results"
        )
    elif analysis_type == "ads_behavior":
        from workers.ads_behavior_analyzer import ADSBehaviorAnalyzer
        analyzer = ADSBehaviorAnalyzer(
            redis_host="localhost",
            redis_port=6379,
            embedding_url=embedding_url,
            vlm_proxy_url=vlm_proxy_url,
            results_dir="/tmp/eval_results"
        )
    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")

    # 加载模板嵌入
    analyzer.load_template_embeddings()

    # 准备输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评测结果
    results: List[EvalResult] = []

    # 处理视频
    video_files = list(Path(video_dir).glob("*.mp4"))
    if limit > 0:
        video_files = video_files[:limit]

    print(f"[INFO] 开始评测 {len(video_files)} 个视频...")

    for i, video_path in enumerate(video_files):
        video_name = video_path.name

        # 获取标注
        ground_truth = labels.get(video_name, labels.get(video_path.stem, False))

        print(f"[{i+1}/{len(video_files)}] 处理: {video_name} (标注={ground_truth})")

        # 评测
        job_id = f"eval_{analysis_type}_{i}"
        result_dict = evaluate_single_video(str(video_path), analyzer, job_id)

        # 判定
        predicted = result_dict["judgment"] == "YES"
        correct = predicted == ground_truth

        result = EvalResult(
            video_path=str(video_path),
            video_name=video_name,
            ground_truth=ground_truth,
            predicted=predicted,
            pred_label=result_dict["judgment"],
            confidence=result_dict["confidence"],
            correct=correct,
            violation_type=result_dict.get("violation_type"),
            reason=result_dict["reason"],
            processing_time=result_dict["processing_time"],
            clip_score=result_dict["clip_score"],
            error=result_dict.get("error")
        )
        results.append(result)

        print(f"    -> 预测={result_dict['judgment']}, 正确={correct}, "
              f"clip_score={result_dict['clip_score']:.3f}, 耗时={result_dict['processing_time']:.1f}s")

    # 计算指标
    tp = sum(1 for r in results if r.ground_truth and r.predicted)
    fp = sum(1 for r in results if not r.ground_truth and r.predicted)
    tn = sum(1 for r in results if not r.ground_truth and not r.predicted)
    fn = sum(1 for r in results if r.ground_truth and not r.predicted)

    total = len(results)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    avg_time = sum(r.processing_time for r in results) / total if total > 0 else 0.0

    summary = EvalSummary(
        analysis_type=analysis_type,
        total_videos=total,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        recall=recall,
        precision=precision,
        f1=f1,
        accuracy=accuracy,
        fpr=fpr,
        processing_time_avg=avg_time,
        timestamp=datetime.now().isoformat()
    )

    # 保存结果
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)

    with open(output_dir / "per_file.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

    # 生成 Markdown 报告
    report = f"""# {analysis_type} 评测报告

## 评测时间
{summary.timestamp}

## 评测指标

| 指标 | 值 |
|------|-----|
| 总视频数 | {total} |
| TP (真阳性) | {tp} |
| FP (假阳性) | {fp} |
| TN (真阴性) | {tn} |
| FN (假阴性) | {fn} |
| **Recall** | **{recall:.4f}** |
| **Precision** | **{precision:.4f}** |
| **F1 Score** | **{f1:.4f}** |
| Accuracy | {accuracy:.4f} |
| FPR | {fpr:.4f} |
| 平均处理时间 | {avg_time:.1f}s |

## FN 案例分析 (漏检)

"""
    fn_cases = [r for r in results if r.ground_truth and not r.predicted]
    for r in fn_cases[:10]:
        report += f"- **{r.video_name}**: clip_score={r.clip_score:.3f}, reason={r.reason}\n"

    report += f"\n## FP 案例分析 (误检)\n\n"
    fp_cases = [r for r in results if not r.ground_truth and r.predicted]
    for r in fp_cases[:10]:
        report += f"- **{r.video_name}**: confidence={r.confidence:.2f}, violation_type={r.violation_type}\n"

    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "="*50)
    print(f"评测完成: {analysis_type}")
    print(f"  Total: {total}, TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    print(f"  输出目录: {output_dir}")
    print("="*50)

    return summary


def main():
    parser = argparse.ArgumentParser(description="多语义检测评测")
    parser.add_argument("--analysis-type", required=True,
                        choices=["mv_violation", "ebike_violation", "ads_behavior"],
                        help="分析类型")
    parser.add_argument("--video-dir", required=True, help="测试视频目录")
    parser.add_argument("--labels-file", required=True, help="标注文件 (JSON)")
    parser.add_argument("--output-dir", default="outputs/multi_semantic_eval",
                        help="输出目录")
    parser.add_argument("--embedding-url", default="http://localhost:8080",
                        help="Embedding 服务地址")
    parser.add_argument("--vlm-proxy-url", default="http://localhost:8001",
                        help="VLM 代理地址")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理视频数量 (0=不限制)")

    args = parser.parse_args()

    run_evaluation(
        analysis_type=args.analysis_type,
        video_dir=args.video_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
