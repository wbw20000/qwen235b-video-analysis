# -*- coding: utf-8 -*-
"""
评测框架核心模块

功能：
- 扫描事故/非事故目录，形成Ground Truth
- 对每个文件调用pipeline
- 收集predict_file输出
- 计算TP/FP/TN/FN + Recall + FPR + Precision
"""
from __future__ import annotations

import os
import glob
import json
import time
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback


@dataclass
class ClipResult:
    """单个clip的评测结果"""
    clip_id: str
    verdict: str  # YES/NO/UNCERTAIN/POST_EVENT_ONLY
    confidence: float
    rank_score: float
    final_score: float
    kept: bool
    keep_reason: str


@dataclass
class PredictResult:
    """文件级预测结果"""
    video_path: str
    predicted_has_accident: bool
    decision_reason: str
    topk: List[ClipResult]
    processing_time: float = 0.0
    error: Optional[str] = None
    # 新增：三态预测标签 (YES/NO/UNCERTAIN)
    pred_label: str = "NO"  # YES/NO/UNCERTAIN
    has_uncertain: bool = False  # 是否包含UNCERTAIN判定
    # 新增：完整pipeline输出（用于result.json.gz dump）
    raw_pipeline_result: Optional[Dict] = None


@dataclass
class EvalResult:
    """评测结果"""
    video_path: str
    ground_truth: bool  # True=事故, False=非事故
    predicted: bool
    correct: bool
    predict_result: Optional[PredictResult] = None


def predict_file(
    video_path: str,
    config: Dict,
    pipeline=None,
    user_query: str = "检测交通事故",
) -> PredictResult:
    """
    文件级预测函数

    Args:
        video_path: 视频文件路径
        config: 配置字典（包含可调参数）
        pipeline: 可选的已初始化pipeline对象
        user_query: 用户查询

    Returns:
        PredictResult: 包含预测结果和详细信息
    """
    from traffic_vlm.pipeline import TrafficVLMPipeline
    from traffic_vlm.config import TrafficVLMConfig

    start_time = time.time()

    try:
        # 构建配置
        vlm_config = TrafficVLMConfig()

        # 应用可调参数
        if "clip_score_threshold" in config:
            vlm_config.vlm.clip_score_threshold = config["clip_score_threshold"]
        if "top_clips" in config:
            vlm_config.vlm.top_clips = config["top_clips"]
        if "skip_low_score_vlm" in config:
            vlm_config.vlm.skip_low_score_vlm = config["skip_low_score_vlm"]
        if "B_full_process" in config:
            vlm_config.rank_score.full_process_bonus_weight = config["B_full_process"]
        if "P_post_event" in config:
            vlm_config.rank_score.post_event_penalty_weight = config["P_post_event"]
        if "min_pre_sec" in config:
            vlm_config.rank_score.pre_roll = config["min_pre_sec"]
        if "min_post_sec" in config:
            vlm_config.rank_score.post_roll = config["min_post_sec"]
        if "pre_roll" in config:
            vlm_config.coverage.pre_roll = config["pre_roll"]
        if "post_roll" in config:
            vlm_config.coverage.post_roll = config["post_roll"]
        if "sampling_fps" in config:
            # 采样帧数 = 时长 * fps，但我们用sampling_frames代替
            pass  # 暂不直接支持fps
        if "fallback_topM" in config:
            # fallback开关：当n_pass_score==0时，强制保留TopM
            pass  # 需要在pipeline中实现
        if "enable_top1_fallback" in config:
            vlm_config.vlm.enable_top1_fallback = config["enable_top1_fallback"]

        # S3阶段配置
        if "enable_s3" in config:
            vlm_config.stage3.enabled = config["enable_s3"]
        if "stage3_weather_prompt_enabled" in config:
            vlm_config.stage3.prompt_injection_enabled = config["stage3_weather_prompt_enabled"]
        if "stage3_roi_enabled" in config:
            vlm_config.stage3.roi_crop_enabled = config["stage3_roi_enabled"]

        # 强制启用YOLO检测（评测必需）
        vlm_config.detector.enabled = True

        # 创建pipeline
        if pipeline is None:
            pipeline = TrafficVLMPipeline(config=vlm_config)

        # 运行pipeline
        result = pipeline.run(
            video_path=video_path,
            user_query=user_query,
            mode="accident",
            generate_report=False,
        )

        # 解析结果
        final_results = result.get("results", [])
        skipped_clips = result.get("skipped_clips", [])

        # 构建topk结果
        topk_results = []
        decision_reason = "无clip通过阈值"

        # 新版判定：基于verdict三态标签
        # pred_label优先级: YES > UNCERTAIN > NO
        pred_label = "NO"
        has_uncertain = False

        for r in final_results:
            clip = r.get("clip", {})
            vlm_output = r.get("vlm_output", {})

            verdict = vlm_output.get("verdict", "NO")
            if verdict is None:
                verdict = "YES" if vlm_output.get("has_accident", False) else "NO"

            confidence = vlm_output.get("confidence", 0.0)
            kept = vlm_output.get("retain_flag", vlm_output.get("kept", False))
            keep_reason = vlm_output.get("retain_reason", vlm_output.get("keep_reason", ""))

            # 基于verdict更新pred_label（优先级: YES > UNCERTAIN > NO）
            if verdict == "YES":
                pred_label = "YES"
                if not decision_reason.startswith("检测到事故"):
                    decision_reason = f"检测到事故: clip={clip.get('clip_id')}, verdict=YES"
            elif verdict == "UNCERTAIN" and pred_label != "YES":
                pred_label = "UNCERTAIN"
                has_uncertain = True
                if not decision_reason.startswith("检测到") and not decision_reason.startswith("疑似"):
                    decision_reason = f"疑似事故: clip={clip.get('clip_id')}, verdict=UNCERTAIN"

            topk_results.append(ClipResult(
                clip_id=clip.get("clip_id", "unknown"),
                verdict=verdict,
                confidence=confidence,
                rank_score=clip.get("final_score", clip.get("clip_score", 0.0)),
                final_score=clip.get("final_score", clip.get("clip_score", 0.0)),
                kept=kept,
                keep_reason=keep_reason,
            ))

        # predicted_has_accident 现在仅在 pred_label=="YES" 时为True（STRICT模式）
        # CONSERVATIVE模式可以在外层通过 pred_label in ("YES", "UNCERTAIN") 判断
        predicted_has_accident = (pred_label == "YES")

        processing_time = time.time() - start_time

        return PredictResult(
            video_path=video_path,
            predicted_has_accident=predicted_has_accident,
            decision_reason=decision_reason,
            topk=topk_results,
            processing_time=processing_time,
            pred_label=pred_label,
            has_uncertain=has_uncertain,
            raw_pipeline_result=result,  # 保存完整pipeline输出
        )

    except Exception as e:
        processing_time = time.time() - start_time
        return PredictResult(
            video_path=video_path,
            predicted_has_accident=False,
            decision_reason=f"处理失败: {str(e)}",
            topk=[],
            processing_time=processing_time,
            error=traceback.format_exc(),
        )


class Evaluator:
    """评测器类"""

    def __init__(
        self,
        acc_dir: str,
        nonacc_dir: str,
        config: Dict,
        output_dir: str = "evaluation_reports",
        subset_per_class: int = 0,  # 0=全量
        seed: int = 42,
        dump_video_results: bool = False,  # 新增：是否保存result.json.gz
    ):
        self.acc_dir = acc_dir
        self.nonacc_dir = nonacc_dir
        self.config = config
        self.output_dir = output_dir
        self.subset_per_class = subset_per_class
        self.seed = seed
        self.dump_video_results = dump_video_results

        # 扫描文件
        self.acc_files = self._scan_mp4(acc_dir)
        self.nonacc_files = self._scan_mp4(nonacc_dir)

        # 如果指定了subset，随机抽样
        if subset_per_class > 0:
            import random
            random.seed(seed)
            if len(self.acc_files) > subset_per_class:
                self.acc_files = random.sample(self.acc_files, subset_per_class)
            if len(self.nonacc_files) > subset_per_class:
                self.nonacc_files = random.sample(self.nonacc_files, subset_per_class)

        print(f"[Evaluator] 事故文件: {len(self.acc_files)}, 非事故文件: {len(self.nonacc_files)}")

    def _scan_mp4(self, directory: str) -> List[str]:
        """扫描目录下的mp4文件（一级目录）"""
        pattern = os.path.join(directory, "*.mp4")
        files = glob.glob(pattern)
        return sorted(files)

    def _dump_video_result(self, video_path: str, pred_result: 'PredictResult', ground_truth: bool):
        """
        保存单个视频的完整result到data/video_results/*.result.json.gz

        Args:
            video_path: 视频路径
            pred_result: 预测结果
            ground_truth: ground truth标签
        """
        import gzip

        # 创建输出目录
        output_dir = "data/video_results"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名（使用视频文件名）
        video_name = os.path.basename(video_path)
        output_file = os.path.join(output_dir, f"{video_name}.result.json.gz")

        # 构建完整result数据
        result_data = {
            "video_info": {
                "video_path": video_path,
                "video_name": video_name,
                "ground_truth": ground_truth,
            },
            "config_snapshot": self.config.copy(),
            "run_metadata": {
                "processing_time": pred_result.processing_time,
                "timestamp": datetime.now().isoformat(),
                "error": pred_result.error,
            },
            "prediction": {
                "pred_label": pred_result.pred_label,
                "predicted_has_accident": pred_result.predicted_has_accident,
                "decision_reason": pred_result.decision_reason,
                "has_uncertain": pred_result.has_uncertain,
            },
            # 完整pipeline输出
            "clips": pred_result.raw_pipeline_result.get("clips", []) if pred_result.raw_pipeline_result else [],
            "results": pred_result.raw_pipeline_result.get("results", []) if pred_result.raw_pipeline_result else [],
            "skipped_clips": pred_result.raw_pipeline_result.get("skipped_clips", []) if pred_result.raw_pipeline_result else [],
            "vlm_stats": pred_result.raw_pipeline_result.get("vlm_stats", {}) if pred_result.raw_pipeline_result else {},
            "templates": pred_result.raw_pipeline_result.get("templates", []) if pred_result.raw_pipeline_result else [],
            "retry_metadata": pred_result.raw_pipeline_result.get("retry_metadata", {}) if pred_result.raw_pipeline_result else {},
            "perf_stats": pred_result.raw_pipeline_result.get("perf_stats", {}) if pred_result.raw_pipeline_result else {},
        }

        # 保存为压缩JSON
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"    ✓ result.json.gz已保存: {output_file}")

    def run(self, eval_id: Optional[str] = None) -> Dict:
        """
        运行评测

        Returns:
            评测结果字典
        """
        if eval_id is None:
            eval_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        eval_dir = os.path.join(self.output_dir, eval_id)
        os.makedirs(eval_dir, exist_ok=True)

        results: List[EvalResult] = []

        # 处理事故文件（正样本）
        print(f"\n[Evaluator] 处理事故文件 ({len(self.acc_files)}个)...")
        for i, video_path in enumerate(self.acc_files):
            print(f"  [{i+1}/{len(self.acc_files)}] {os.path.basename(video_path)}")
            pred_result = predict_file(video_path, self.config)

            # 保存result.json.gz（如果启用）
            if self.dump_video_results:
                self._dump_video_result(video_path, pred_result, ground_truth=True)

            results.append(EvalResult(
                video_path=video_path,
                ground_truth=True,  # 事故
                predicted=pred_result.predicted_has_accident,
                correct=pred_result.predicted_has_accident == True,
                predict_result=pred_result,
            ))

            print(f"    预测: {'事故' if pred_result.predicted_has_accident else '非事故'} "
                  f"({'正确' if pred_result.predicted_has_accident else '漏报'}) "
                  f"- {pred_result.decision_reason[:50]}")

        # 处理非事故文件（负样本）
        print(f"\n[Evaluator] 处理非事故文件 ({len(self.nonacc_files)}个)...")
        for i, video_path in enumerate(self.nonacc_files):
            print(f"  [{i+1}/{len(self.nonacc_files)}] {os.path.basename(video_path)}")
            pred_result = predict_file(video_path, self.config)

            # 保存result.json.gz（如果启用）
            if self.dump_video_results:
                self._dump_video_result(video_path, pred_result, ground_truth=False)

            results.append(EvalResult(
                video_path=video_path,
                ground_truth=False,  # 非事故
                predicted=pred_result.predicted_has_accident,
                correct=pred_result.predicted_has_accident == False,
                predict_result=pred_result,
            ))

            print(f"    预测: {'事故' if pred_result.predicted_has_accident else '非事故'} "
                  f"({'误报' if pred_result.predicted_has_accident else '正确'}) "
                  f"- {pred_result.decision_reason[:50]}")

        # 计算指标（两套口径）
        from .metrics import compute_metrics_dual
        metrics_dual = compute_metrics_dual(results)

        # 默认使用STRICT模式的指标
        metrics = metrics_dual["strict"]

        # 生成报告（同时包含两套指标）
        self._generate_report(eval_dir, eval_id, results, metrics_dual)

        return {
            "eval_id": eval_id,
            "eval_dir": eval_dir,
            "metrics": metrics,
            "metrics_dual": metrics_dual,
            "results": results,
        }

    def _generate_report(
        self,
        eval_dir: str,
        eval_id: str,
        results: List[EvalResult],
        metrics_dual: Dict,
    ):
        """生成评测报告（同时输出STRICT和CONSERVATIVE两套指标）"""
        # 获取两套指标
        strict = metrics_dual.get("strict", metrics_dual)
        conservative = metrics_dual.get("conservative", metrics_dual)

        # summary.json
        summary = {
            "eval_id": eval_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "dataset": {
                "acc_dir": self.acc_dir,
                "nonacc_dir": self.nonacc_dir,
                "acc_count": len(self.acc_files),
                "nonacc_count": len(self.nonacc_files),
            },
            "metrics": strict,  # 默认使用STRICT
            "metrics_dual": metrics_dual,
        }

        with open(os.path.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # per_file.json
        per_file = []
        for r in results:
            item = {
                "video_path": r.video_path,
                "video_name": os.path.basename(r.video_path),
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "correct": r.correct,
            }
            if r.predict_result:
                item["decision_reason"] = r.predict_result.decision_reason
                item["processing_time"] = r.predict_result.processing_time
                item["error"] = r.predict_result.error
                item["topk_count"] = len(r.predict_result.topk)
                item["pred_label"] = r.predict_result.pred_label
                item["has_uncertain"] = r.predict_result.has_uncertain
            per_file.append(item)

        with open(os.path.join(eval_dir, "per_file.json"), "w", encoding="utf-8") as f:
            json.dump(per_file, f, ensure_ascii=False, indent=2)

        # false_positives.md (误报) - 基于STRICT模式
        fps = [r for r in results if r.ground_truth == False and r.predicted == True]
        with open(os.path.join(eval_dir, "false_positives.md"), "w", encoding="utf-8") as f:
            f.write(f"# 误报列表 (FP={len(fps)})\n\n")
            for i, r in enumerate(fps[:10]):
                f.write(f"## {i+1}. {os.path.basename(r.video_path)}\n")
                f.write(f"- 路径: `{r.video_path}`\n")
                if r.predict_result:
                    f.write(f"- 原因: {r.predict_result.decision_reason}\n")
                    f.write(f"- pred_label: {r.predict_result.pred_label}\n")
                f.write("\n")

        # false_negatives.md (漏报)
        fns = [r for r in results if r.ground_truth == True and r.predicted == False]
        with open(os.path.join(eval_dir, "false_negatives.md"), "w", encoding="utf-8") as f:
            f.write(f"# 漏报列表 (FN={len(fns)})\n\n")
            for i, r in enumerate(fns[:10]):
                f.write(f"## {i+1}. {os.path.basename(r.video_path)}\n")
                f.write(f"- 路径: `{r.video_path}`\n")
                if r.predict_result:
                    f.write(f"- 原因: {r.predict_result.decision_reason}\n")
                    f.write(f"- pred_label: {r.predict_result.pred_label}\n")
                f.write("\n")

        # report.md - 同时输出两套指标
        with open(os.path.join(eval_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(f"# 评测报告\n\n")
            f.write(f"- 评测ID: {eval_id}\n")
            f.write(f"- 时间: {datetime.now().isoformat()}\n\n")

            f.write("## 数据集\n")
            f.write(f"- 事故目录: `{self.acc_dir}`\n")
            f.write(f"- 非事故目录: `{self.nonacc_dir}`\n")
            f.write(f"- 事故文件数: {len(self.acc_files)}\n")
            f.write(f"- 非事故文件数: {len(self.nonacc_files)}\n\n")

            # 预测标签分布
            label_dist = strict.get("label_dist", {})
            f.write("## 预测标签分布\n")
            f.write(f"- YES (确定事故): {label_dist.get('YES', 0)}\n")
            f.write(f"- NO (非事故): {label_dist.get('NO', 0)}\n")
            f.write(f"- UNCERTAIN (待复核): {label_dist.get('UNCERTAIN', 0)}\n\n")

            # STRICT模式指标（主要）
            f.write("## 评测指标 (STRICT模式 - 推荐)\n")
            f.write("*仅 YES 计为事故，UNCERTAIN 作为 abstain 不计入*\n\n")
            f.write(f"- **Recall (TPR)**: {strict['recall']:.4f}\n")
            f.write(f"- **FPR**: {strict['fpr']:.4f}\n")
            f.write(f"- **Precision**: {strict['precision']:.4f}\n")
            f.write(f"- **F1**: {strict['f1']:.4f}\n")
            f.write(f"- **Accuracy**: {strict['accuracy']:.4f}\n")
            f.write(f"- **Abstain Rate**: {strict['abstain_rate']:.4f} ({strict['abstain']}个样本)\n\n")

            f.write("### 混淆矩阵 (STRICT)\n")
            f.write(f"- TP (正确识别事故): {strict['tp']}\n")
            f.write(f"- FP (误报): {strict['fp']}\n")
            f.write(f"- TN (正确识别非事故): {strict['tn']}\n")
            f.write(f"- FN (漏报): {strict['fn']}\n\n")

            # CONSERVATIVE模式指标
            f.write("## 评测指标 (CONSERVATIVE模式)\n")
            f.write("*YES + UNCERTAIN 都计为事故*\n\n")
            f.write(f"- **Recall (TPR)**: {conservative['recall']:.4f}\n")
            f.write(f"- **FPR**: {conservative['fpr']:.4f}\n")
            f.write(f"- **Precision**: {conservative['precision']:.4f}\n")
            f.write(f"- **F1**: {conservative['f1']:.4f}\n")
            f.write(f"- **Accuracy**: {conservative['accuracy']:.4f}\n\n")

            f.write("### 混淆矩阵 (CONSERVATIVE)\n")
            f.write(f"- TP (正确识别事故): {conservative['tp']}\n")
            f.write(f"- FP (误报): {conservative['fp']}\n")
            f.write(f"- TN (正确识别非事故): {conservative['tn']}\n")
            f.write(f"- FN (漏报): {conservative['fn']}\n\n")

            f.write("## 配置参数\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2, ensure_ascii=False))
            f.write("\n```\n")

        # 生成 casebook.md 和 decision_trace.json
        self._generate_casebook(eval_dir, results)
        self._generate_decision_trace(eval_dir, results)

        print(f"\n[Evaluator] 报告已保存到: {eval_dir}")

    def _generate_casebook(self, eval_dir: str, results: List[EvalResult]):
        """生成 casebook.md - FN/FP 详细案例分析"""
        import gzip

        # 筛选 FN 和 FP
        fns = [r for r in results if r.ground_truth == True and r.predicted == False]
        fps = [r for r in results if r.ground_truth == False and r.predicted == True]

        with open(os.path.join(eval_dir, "casebook.md"), "w", encoding="utf-8") as f:
            f.write("# Casebook - FN/FP 案例分析\n\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n\n")
            f.write(f"- FN (漏报): {len(fns)} 个\n")
            f.write(f"- FP (误报): {len(fps)} 个\n\n")

            # FN 案例
            f.write("---\n\n")
            f.write("# False Negatives (漏报)\n\n")
            for i, r in enumerate(fns):
                video_name = os.path.basename(r.video_path)
                f.write(f"## FN-{i+1}: {video_name}\n\n")
                f.write(f"- **路径**: `{r.video_path}`\n")
                f.write(f"- **Ground Truth**: 事故\n")
                f.write(f"- **Prediction**: {r.predict_result.pred_label if r.predict_result else 'N/A'}\n")
                f.write(f"- **Decision Reason**: {r.predict_result.decision_reason if r.predict_result else 'N/A'}\n\n")

                # 尝试读取 result.json.gz 获取详细信息
                result_file = os.path.join("data/video_results", f"{video_name}.result.json.gz")
                if os.path.exists(result_file):
                    try:
                        with gzip.open(result_file, 'rt', encoding='utf-8') as rf:
                            result_data = json.load(rf)

                        # Clip 分析表格
                        results_list = result_data.get("results", [])
                        if results_list:
                            f.write("### Clip 分析\n\n")
                            f.write("| clip_id | base_score | final_score | VLM verdict | confidence |\n")
                            f.write("|---------|------------|-------------|-------------|------------|\n")
                            for clip_result in results_list:
                                clip = clip_result.get("clip", {})
                                vlm = clip_result.get("vlm_output", {})
                                clip_id = clip.get("clip_id", "N/A")
                                base_score = clip.get("clip_score", 0)
                                final_score = clip.get("final_score", 0)
                                # 获取最终verdict
                                verdict = vlm.get("verdict", "N/A")
                                conf = vlm.get("confidence", 0)
                                f.write(f"| {clip_id} | {base_score:.4f} | {final_score:.4f} | {verdict} | {conf:.2f} |\n")
                            f.write("\n")

                        # VLM 响应详情
                        if results_list:
                            vlm = results_list[0].get("vlm_output", {})
                            f.write("### VLM 响应\n\n")
                            s1 = vlm.get("stage1_result", {})
                            s2 = vlm.get("stage2_result", {})
                            if s1:
                                f.write(f"- **S1**: verdict={s1.get('verdict')}, confidence={s1.get('confidence', 0):.2f}\n")
                            if s2:
                                f.write(f"- **S2**: verdict={s2.get('verdict')}, confidence={s2.get('confidence', 0):.2f}\n")
                            f.write(f"- **Final**: {vlm.get('verdict')}\n")
                            summary = s2.get("text_summary") or s1.get("text_summary") or vlm.get("text_summary", "")
                            if summary:
                                f.write(f"- **Summary**: \"{summary[:200]}...\"\n")
                            f.write("\n")

                    except Exception as e:
                        f.write(f"*无法读取详细信息: {e}*\n\n")
                else:
                    f.write("*未找到 result.json.gz 文件*\n\n")

            # FP 案例
            f.write("---\n\n")
            f.write("# False Positives (误报)\n\n")
            for i, r in enumerate(fps):
                video_name = os.path.basename(r.video_path)
                f.write(f"## FP-{i+1}: {video_name}\n\n")
                f.write(f"- **路径**: `{r.video_path}`\n")
                f.write(f"- **Ground Truth**: 非事故\n")
                f.write(f"- **Prediction**: {r.predict_result.pred_label if r.predict_result else 'N/A'}\n")
                f.write(f"- **Decision Reason**: {r.predict_result.decision_reason if r.predict_result else 'N/A'}\n\n")

                # 尝试读取详细信息
                result_file = os.path.join("data/video_results", f"{video_name}.result.json.gz")
                if os.path.exists(result_file):
                    try:
                        with gzip.open(result_file, 'rt', encoding='utf-8') as rf:
                            result_data = json.load(rf)

                        results_list = result_data.get("results", [])
                        if results_list:
                            f.write("### Clip 分析\n\n")
                            f.write("| clip_id | base_score | final_score | VLM verdict | confidence |\n")
                            f.write("|---------|------------|-------------|-------------|------------|\n")
                            for clip_result in results_list:
                                clip = clip_result.get("clip", {})
                                vlm = clip_result.get("vlm_output", {})
                                clip_id = clip.get("clip_id", "N/A")
                                base_score = clip.get("clip_score", 0)
                                final_score = clip.get("final_score", 0)
                                verdict = vlm.get("verdict", "N/A")
                                conf = vlm.get("confidence", 0)
                                f.write(f"| {clip_id} | {base_score:.4f} | {final_score:.4f} | {verdict} | {conf:.2f} |\n")
                            f.write("\n")

                            # 找到返回YES的clip详细信息
                            for clip_result in results_list:
                                vlm = clip_result.get("vlm_output", {})
                                if vlm.get("verdict") == "YES":
                                    f.write("### 误判 VLM 响应\n\n")
                                    summary = vlm.get("text_summary", "")
                                    f.write(f"- **Summary**: \"{summary[:300]}...\"\n\n")
                                    break

                    except Exception as e:
                        f.write(f"*无法读取详细信息: {e}*\n\n")
                else:
                    f.write("*未找到 result.json.gz 文件*\n\n")

        print(f"    [OK] casebook.md 已生成")

    def _generate_decision_trace(self, eval_dir: str, results: List[EvalResult]):
        """生成 decision_trace.json - 每个视频的决策追踪链"""
        import gzip

        traces = []

        for r in results:
            video_name = os.path.basename(r.video_path)
            trace = {
                "video_name": video_name,
                "video_path": r.video_path,
                "ground_truth": r.ground_truth,
                "predicted": r.predicted,
                "correct": r.correct,
                "pred_label": r.predict_result.pred_label if r.predict_result else None,
                "decision_reason": r.predict_result.decision_reason if r.predict_result else None,
                "processing_time": r.predict_result.processing_time if r.predict_result else 0,
                "error": r.predict_result.error if r.predict_result else None,
                "decision_chain": []
            }

            # 尝试读取详细结果构建决策链
            result_file = os.path.join("data/video_results", f"{video_name}.result.json.gz")
            if os.path.exists(result_file):
                try:
                    with gzip.open(result_file, 'rt', encoding='utf-8') as rf:
                        result_data = json.load(rf)

                    clips = result_data.get("clips", [])
                    results_list = result_data.get("results", [])
                    skipped = result_data.get("skipped_clips", [])
                    config = result_data.get("config_snapshot", {})
                    threshold = config.get("clip_score_threshold", 0.35)

                    # 阶段1: clip生成
                    clip_scores = [c.get("clip_score", 0) for c in clips]
                    trace["decision_chain"].append({
                        "stage": "clip_generation",
                        "clips_count": len(clips),
                        "scores": clip_scores[:5]  # 只保留前5个
                    })

                    # 阶段2: 覆盖度评分
                    final_scores = [c.get("final_score", 0) for c in clips]
                    trace["decision_chain"].append({
                        "stage": "coverage_scoring",
                        "final_scores": final_scores[:5],
                        "threshold": threshold
                    })

                    # 阶段3: 阈值过滤
                    trace["decision_chain"].append({
                        "stage": "threshold_filter",
                        "passed": len(results_list),
                        "skipped": len(skipped)
                    })

                    # 阶段4-6: VLM分析
                    for clip_result in results_list[:1]:  # 只记录第一个
                        vlm = clip_result.get("vlm_output", {})
                        s1 = vlm.get("stage1_result", {})
                        s2 = vlm.get("stage2_result", {})

                        if s1:
                            trace["decision_chain"].append({
                                "stage": "vlm_s1",
                                "verdict": s1.get("verdict"),
                                "confidence": s1.get("confidence", 0)
                            })

                        if vlm.get("escalated"):
                            trace["decision_chain"].append({
                                "stage": "escalation",
                                "triggered": True,
                                "reason": vlm.get("escalation_reason", "unknown")
                            })

                        if s2:
                            trace["decision_chain"].append({
                                "stage": "vlm_s2",
                                "verdict": s2.get("verdict"),
                                "confidence": s2.get("confidence", 0)
                            })

                        # 最终决策
                        trace["decision_chain"].append({
                            "stage": "final_decision",
                            "verdict": vlm.get("verdict"),
                            "logic": "S1=NO takes precedence" if s1.get("verdict") == "NO" and s2.get("verdict") == "UNCERTAIN" else "standard"
                        })

                except Exception as e:
                    trace["decision_chain"].append({
                        "stage": "error",
                        "message": str(e)
                    })

            traces.append(trace)

        # 保存
        with open(os.path.join(eval_dir, "decision_trace.json"), "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

        print(f"    [OK] decision_trace.json 已生成")
