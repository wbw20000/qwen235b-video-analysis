"""
评分分解输出模块 (Scorecard)

为每个clip生成详细的分数分解，便于审计和调试。
"""

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def safe_serialize(obj: Any) -> Any:
    """递归转换numpy/tensor等类型为Python原生类型"""
    if HAS_NUMPY:
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass

    if isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_serialize(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return str(obj)


class ScorecardBuilder:
    """分数分解输出器（语义对齐版）"""

    # 当前使用的权重常量
    WEIGHTS = {
        # final_score 公式权重 (file_event_locator.py:605)
        "w_base": 1.0,
        "w_coverage": 0.3,
        "w_late": 0.02,

        # rank_score 公式权重（单一排序主分）
        # rank_score = final_score + B*full_process_score - P*post_event_score + verdict_bonus
        "B_full_process": 0.20,            # full_process_bonus权重
        "P_post_event": 0.20,              # post_event_penalty权重
        "confirm_verdict_bonus": 0.10,     # verdict=YES加分
        "uncertain_verdict_bonus": 0.05,   # verdict=UNCERTAIN加分

        # t0_validity 各项证据权重
        "t0_min_dist": 0.25,
        "t0_dist_drop": 0.30,
        "t0_velocity": 0.25,
        "t0_iou": 0.10,
        "t0_jitter": 0.10,
    }

    # 当前使用的阈值
    THRESHOLDS = {
        "clip_score_threshold": 0.35,
        "skip_low_score_vlm": True,
        "top_clips": 3,
        "validity_threshold": 0.3,
        "risk_peak_threshold": 0.25,
        "roi_median_threshold": 80.0,
        "pre_roll": 8.0,
        "post_roll": 12.0,
    }

    def __init__(self, output_dir: str, run_id: str = None):
        """
        Args:
            output_dir: 报告输出目录 (e.g., "reports")
            run_id: 运行ID，不指定则自动生成
        """
        self.run_id = run_id or self._generate_run_id()
        self.output_dir = Path(output_dir) / self.run_id
        self.clips_dir = self.output_dir / "clips"
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.scorecards: List[Dict] = []

    def _generate_run_id(self) -> str:
        import subprocess
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            git_hash = result.stdout.strip() if result.returncode == 0 else "nogit"
        except Exception:
            git_hash = "nogit"
        return f"{ts}_{git_hash}"

    def add_candidate(
        self,
        candidate: Dict,
        rank: int = 0,
        vlm_result: Optional[Dict] = None,
    ) -> Dict:
        """
        为单个候选生成scorecard

        Args:
            candidate: 候选dict（包含所有分数字段）
            rank: 排名
            vlm_result: VLM结果（如果调用了）

        Returns:
            生成的scorecard dict
        """
        clip_id = candidate.get("clip_id", "unknown")

        # 构建scorecard
        scorecard = {
            "clip_id": clip_id,
            "source_path": candidate.get("video_path", ""),
            "duration_sec": candidate.get("duration", 0),
            "time_range": {
                "start": candidate.get("start_time", 0),
                "end": candidate.get("end_time", 0),
            },

            "scores": {
                # 基础分数
                "clip_score": candidate.get("clip_score", 0),
                "accident_score": candidate.get("accident_score", 0),
                "base_score": candidate.get("base_score", candidate.get("clip_score", 0)),
                "risk_peak": candidate.get("risk_peak", 0),

                # t0相关
                "t0_sec": candidate.get("t0", 0),
                "t0_fallback": candidate.get("t0_fallback", False),
                "t0_validity": candidate.get("t0_validity", 0),
                "t0_validity_reason": candidate.get("validity_reason", ""),

                # 旧版覆盖度（兼容）
                "pre_ok": candidate.get("pre_ok", 0),
                "post_ok": candidate.get("post_ok", 0),
                "coverage_score_raw": candidate.get("coverage_raw", 0),
                "coverage_effective": candidate.get("coverage_effective", 0),
                "late_start_penalty": candidate.get("late_start_penalty", 0),

                # ===== 新版语义对齐评分 =====
                # 三因子分数: pre_score * impact_score * post_score
                "pre_score": candidate.get("pre_score", 0),
                "impact_score": candidate.get("impact_score", 0),
                "post_score": candidate.get("post_score", 0),

                # 语义对齐分数（根据verdict）
                "full_process_score": candidate.get("full_process_score", 0),
                "post_event_score": candidate.get("post_event_score", 0),

                # 奖惩分数
                "full_process_bonus": candidate.get("full_process_bonus", 0),
                "post_event_penalty": candidate.get("post_event_penalty", 0),
                "verdict_bonus": candidate.get("verdict_bonus", 0),

                # 最终分数
                "final_score": candidate.get("final_score", 0),
                "rank_score": candidate.get("rank_score", 0),
            },

            "filters": {
                "passed_clip_score_threshold": candidate.get("filter_status") == "PASSED",
                "skipped_reason": candidate.get("skip_reason", ""),
                "topn_selected": candidate.get("filter_status") != "NOT_SELECTED",
                "vlm_called": vlm_result is not None,
            },

            "vlm": {
                "verdict": vlm_result.get("verdict") if vlm_result else None,
                "confidence": vlm_result.get("confidence", 0) if vlm_result else None,
                "evidence_frames": vlm_result.get("evidence_frames", []) if vlm_result else [],
                "impact_description": vlm_result.get("text_summary", "") if vlm_result else "",
            },

            "decision": {
                "rank": rank,
                "kept": candidate.get("kept", False),
                "keep_reason": candidate.get("keep_reason", ""),
            },

            "formula_snapshot": {
                # 旧版公式（兼容）
                "final_score_formula": "final = base + 0.3*cover_eff - 0.02*late_penalty",
                "coverage_formula": "cover_eff = t0_validity * pre_ok * post_ok",

                # ===== 新版语义对齐公式 =====
                "rank_score_formula": "rank_score = final + B*full_process_score - P*post_event_score + verdict_bonus",
                "full_process_score_formula": "full_process_score = pre_score * impact_score * post_score (仅当verdict=YES)",
                "pre_score_formula": "pre_score = min(1.0, t0 / pre_roll)",
                "impact_score_formula": "impact_score = t0_validity",
                "post_score_formula": "post_score = min(1.0, (duration - t0) / post_roll)",

                # 语义约束
                "semantic_constraints": {
                    "YES": "full_process_score=计算值, post_event_score=0",
                    "POST_EVENT_ONLY": "full_process_score=0, post_event_score=1",
                    "UNCERTAIN": "full_process_score=0, post_event_score=0",
                    "NO": "full_process_score=0, post_event_score=0",
                },

                "weights": self.WEIGHTS.copy(),
                "thresholds": self.THRESHOLDS.copy(),
            },
        }

        self.scorecards.append(scorecard)

        # 写入单独的scorecard.json
        self._write_clip_scorecard(clip_id, scorecard)

        return scorecard

    def _write_clip_scorecard(self, clip_id: str, scorecard: Dict):
        """写入单个clip的scorecard.json"""
        clip_dir = self.clips_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        scorecard_file = clip_dir / "scorecard.json"
        with open(scorecard_file, "w", encoding="utf-8") as f:
            json.dump(safe_serialize(scorecard), f, ensure_ascii=False, indent=2)

    def generate_summary(self) -> Dict:
        """
        生成scoring_summary.json

        Returns:
            summary dict
        """
        if not self.scorecards:
            return {"error": "no scorecards"}

        # 统计
        total = len(self.scorecards)
        vlm_called = sum(1 for s in self.scorecards if s["filters"]["vlm_called"])
        passed_threshold = sum(1 for s in self.scorecards if s["filters"]["passed_clip_score_threshold"])
        skipped = sum(1 for s in self.scorecards if not s["filters"]["passed_clip_score_threshold"])
        kept = sum(1 for s in self.scorecards if s["decision"]["kept"])

        # 分数分布
        final_scores = [s["scores"]["final_score"] for s in self.scorecards]
        cover_effs = [s["scores"]["coverage_effective"] for s in self.scorecards]
        clip_scores = [s["scores"]["clip_score"] for s in self.scorecards]

        def distribution_stats(values: List[float]) -> Dict:
            if not values:
                return {"min": 0, "median": 0, "p90": 0, "max": 0}
            sorted_v = sorted(values)
            n = len(sorted_v)
            return {
                "min": sorted_v[0],
                "median": sorted_v[n // 2],
                "p90": sorted_v[int(n * 0.9)] if n > 1 else sorted_v[0],
                "max": sorted_v[-1],
            }

        # Near Miss: 被过滤但最接近阈值的clips
        threshold = self.THRESHOLDS["clip_score_threshold"]
        near_miss = [
            {"clip_id": s["clip_id"], "clip_score": s["scores"]["clip_score"]}
            for s in self.scorecards
            if not s["filters"]["passed_clip_score_threshold"] and s["scores"]["clip_score"] > 0
        ]
        near_miss.sort(key=lambda x: x["clip_score"], reverse=True)
        near_miss = near_miss[:10]

        summary = {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "total_clips": total,
                "vlm_called": vlm_called,
                "passed_threshold": passed_threshold,
                "skipped_by_threshold": skipped,
                "kept": kept,
            },
            "distributions": {
                "final_score": distribution_stats(final_scores),
                "coverage_effective": distribution_stats(cover_effs),
                "clip_score": distribution_stats(clip_scores),
            },
            "near_miss_top10": near_miss,
            "thresholds_used": self.THRESHOLDS.copy(),
            "weights_used": self.WEIGHTS.copy(),
        }

        # 写入文件
        summary_file = self.output_dir / "scoring_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(safe_serialize(summary), f, ensure_ascii=False, indent=2)

        return summary

    def print_ranking_table(self):
        """
        在控制台打印分数表（rank_score版）

        格式:
        rank | clip_id | verdict | t0 | pre | imp | post | full_p | post_e | final | bonus | penalty | rank_score | kept
        """
        print("\n" + "=" * 160)
        print("Scorecard Ranking Table (rank_score版)")
        print("=" * 160)

        header = (
            f"{'rank':>4} | {'clip_id':<16} | {'verdict':<16} | {'t0':>5} | "
            f"{'pre':>5} | {'imp':>5} | {'post':>5} | {'full_p':>6} | {'post_e':>6} | "
            f"{'final':>7} | {'bonus':>6} | {'penalty':>7} | {'rank_score':>10} | {'kept':>4}"
        )
        print(header)
        print("-" * 160)

        # 按rank_score降序排序（与file_event_locator一致）
        sorted_cards = sorted(
            self.scorecards,
            key=lambda x: x["scores"].get("rank_score", 0),
            reverse=True
        )

        for i, s in enumerate(sorted_cards, 1):
            clip_id = s["clip_id"][:16]
            verdict = (s["vlm"]["verdict"] or "-")[:16]
            t0 = s["scores"]["t0_sec"]
            pre = s["scores"].get("pre_score", 0)
            imp = s["scores"].get("impact_score", 0)
            post = s["scores"].get("post_score", 0)
            full_p = s["scores"].get("full_process_score", 0)
            post_e = s["scores"].get("post_event_score", 0)
            final = s["scores"]["final_score"]
            bonus = s["scores"].get("full_process_bonus", 0)
            penalty = s["scores"].get("post_event_penalty", 0)
            rank_score = s["scores"].get("rank_score", 0)
            kept = "YES" if s["decision"]["kept"] else "NO"

            row = (
                f"{i:>4} | {clip_id:<16} | {verdict:<16} | {t0:>5.1f} | "
                f"{pre:>5.2f} | {imp:>5.2f} | {post:>5.2f} | {full_p:>6.3f} | {post_e:>6.1f} | "
                f"{final:>7.4f} | {bonus:>6.3f} | {penalty:>7.3f} | {rank_score:>10.4f} | {kept:>4}"
            )
            print(row)

        print("=" * 160)


def generate_scorecards_from_candidates(
    candidates: List[Dict],
    output_dir: str = "reports",
    run_id: str = None,
    vlm_results: Dict[str, Dict] = None,
    thresholds: Dict = None,
) -> str:
    """
    从候选列表生成scorecards的便捷函数

    Args:
        candidates: 候选列表（已排序）
        output_dir: 报告输出目录
        run_id: 运行ID
        vlm_results: clip_id -> vlm_result的映射
        thresholds: 阈值配置覆盖

    Returns:
        报告目录路径
    """
    builder = ScorecardBuilder(output_dir, run_id)

    if thresholds:
        builder.THRESHOLDS.update(thresholds)

    vlm_results = vlm_results or {}

    for rank, candidate in enumerate(candidates, 1):
        clip_id = candidate.get("clip_id", "unknown")
        vlm_result = vlm_results.get(clip_id)
        builder.add_candidate(candidate, rank, vlm_result)

    builder.generate_summary()
    builder.print_ranking_table()

    return str(builder.output_dir)
