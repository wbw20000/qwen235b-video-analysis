"""
事故分析自动诊断报告系统

每次pipeline运行后自动生成：
- report.md: 人类可读的诊断报告
- report.json: 机器可读的完整报告
- summary.json: 简要摘要
- stage_stats.json: 各阶段统计
- config_snapshot.yaml: 配置快照
- env.txt: 环境信息
- clips/<clip_id>/: 证据目录
"""

import json
import os
import subprocess
import sys
import platform
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ============ JSON安全序列化 ============

def safe_json_serialize(obj: Any) -> Any:
    """递归转换numpy/tensor等类型为Python原生类型，避免json.dump崩溃"""
    if HAS_NUMPY:
        import numpy as np
        if isinstance(obj, (np.bool_, )):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    # PyTorch tensor
    if hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(obj, 'tolist') and callable(obj.tolist):
        try:
            return obj.tolist()
        except (ValueError, RuntimeError):
            pass

    # 基本类型
    if isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float, str, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(i) for i in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, '__dict__'):
        return safe_json_serialize(vars(obj))
    else:
        # 兜底：转字符串
        return str(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全的json.dumps，自动处理numpy等类型"""
    return json.dumps(safe_json_serialize(obj), **kwargs)


def safe_json_dump(obj: Any, fp, **kwargs):
    """安全的json.dump，自动处理numpy等类型"""
    json.dump(safe_json_serialize(obj), fp, **kwargs)


# ============ 运行上下文 ============

@dataclass
class StageStats:
    """单阶段统计"""
    name: str
    duration_sec: float = 0.0
    success: bool = True
    input_count: int = 0
    output_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunContext:
    """运行上下文，收集pipeline运行信息"""
    run_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    mode: str = ""  # "accident" | "violation"
    camera_id: str = ""
    date_str: str = ""

    # 配置快照
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # 各阶段统计
    stages: Dict[str, StageStats] = field(default_factory=dict)

    # 关键计数
    n_source_videos: int = 0
    n_clips_cut: int = 0
    n_preprocessed: int = 0
    n_pass_score: int = 0
    n_vlm_analyzed: int = 0
    n_kept: int = 0
    n_topk: int = 0

    # 阈值
    clip_score_threshold: float = 0.0
    skip_low_score_vlm: bool = True
    top_clips: int = 3

    def generate_run_id(self) -> str:
        """生成run_id: UTC时间戳 + git短hash"""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        git_hash = self._get_git_short_hash()
        self.run_id = f"{ts}_{git_hash}"
        return self.run_id

    def _get_git_short_hash(self) -> str:
        """获取git短hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "nogit"

    def add_stage(self, name: str, **kwargs) -> StageStats:
        """添加阶段统计"""
        stage = StageStats(name=name, **kwargs)
        self.stages[name] = stage
        return stage

    def get_total_duration(self) -> float:
        """获取总耗时"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(s.duration_sec for s in self.stages.values())

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_sec": self.get_total_duration(),
            "mode": self.mode,
            "camera_id": self.camera_id,
            "date_str": self.date_str,
            "counts": {
                "n_source_videos": self.n_source_videos,
                "n_clips_cut": self.n_clips_cut,
                "n_preprocessed": self.n_preprocessed,
                "n_pass_score": self.n_pass_score,
                "n_vlm_analyzed": self.n_vlm_analyzed,
                "n_kept": self.n_kept,
                "n_topk": self.n_topk,
            },
            "thresholds": {
                "clip_score_threshold": self.clip_score_threshold,
                "skip_low_score_vlm": self.skip_low_score_vlm,
                "top_clips": self.top_clips,
            },
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }


# ============ Clip结果 ============

@dataclass
class ClipResult:
    """单个clip的分析结果"""
    clip_id: str
    video_path: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # 分数
    clip_score: float = 0.0
    accident_score: float = 0.0
    final_score: float = 0.0
    coverage_score: float = 0.0
    coverage_effective: float = 0.0  # 有效覆盖度
    t0: float = 0.0                  # 碰撞时刻估计
    t0_validity: float = 0.0
    rank_score: float = 0.0          # 唯一排序主分
    rank: int = 0                    # 排名

    # 过滤状态
    filter_status: str = "UNKNOWN"  # PASSED | SKIPPED | NOT_SELECTED
    skip_reason: str = ""

    # VLM结果
    vlm_verdict: str = ""  # YES | NO | POST_EVENT_ONLY | UNCERTAIN
    vlm_confidence: float = 0.0
    kept: bool = False
    keep_reason: str = ""

    # 证据路径
    evidence_dir: str = ""
    frames_raw: List[str] = field(default_factory=list)
    frames_annotated: List[str] = field(default_factory=list)
    vlm_request_path: str = ""
    vlm_response_path: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


# ============ 短路诊断 ============

@dataclass
class RunGate:
    """运行门禁检查点"""
    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    message: str = ""
    severity: str = "INFO"  # INFO | WARN | FAIL

    def to_dict(self) -> Dict:
        return asdict(self)


class ShortCircuitDiagnostics:
    """短路诊断器：分析为何没有clip进入VLM"""

    def __init__(self, context: RunContext, clip_results: List[ClipResult]):
        self.context = context
        self.clip_results = clip_results
        self.gates: List[RunGate] = []

    def run_diagnostics(self) -> Tuple[str, List[RunGate]]:
        """运行诊断，返回 (总体结论, 门禁列表)"""
        self.gates = []

        # Gate 1: 是否有源视频
        self._check_source_videos()

        # Gate 2: 是否成功剪辑clips
        self._check_clips_cut()

        # Gate 3: Top-K选择
        self._check_topk_selection()

        # Gate 4: clip_score阈值过滤
        self._check_score_threshold()

        # Gate 5: VLM输入数量
        self._check_vlm_input()

        # Gate 6: VLM verdict分布
        self._check_vlm_verdicts()

        # 计算总体结论
        conclusion = self._compute_conclusion()

        return conclusion, self.gates

    def _check_source_videos(self):
        n = self.context.n_source_videos
        gate = RunGate(
            name="source_videos",
            passed=n > 0,
            value=n,
            threshold=">0",
            severity="FAIL" if n == 0 else "INFO"
        )
        if n == 0:
            gate.message = "无源视频输入"
        else:
            gate.message = f"源视频数: {n}"
        self.gates.append(gate)

    def _check_clips_cut(self):
        n = self.context.n_clips_cut
        gate = RunGate(
            name="clips_cut",
            passed=n > 0,
            value=n,
            threshold=">0",
            severity="FAIL" if n == 0 else "INFO"
        )
        if n == 0:
            gate.message = "剪辑阶段未生成任何clip"
        else:
            gate.message = f"成功剪辑: {n} clips"
        self.gates.append(gate)

    def _check_topk_selection(self):
        """检查Top-K选择"""
        n_cut = self.context.n_clips_cut
        n_preprocess = self.context.n_preprocessed
        top_k = self.context.top_clips

        gate = RunGate(
            name="topk_selection",
            passed=True,
            value=n_preprocess,
            threshold=f"top_clips={top_k}",
            severity="INFO"
        )

        if n_preprocess < n_cut:
            gate.message = f"Top-K截断: {n_cut} clips -> 只预处理前 {n_preprocess} 个 (top_clips={top_k})"
            if n_cut > top_k:
                gate.severity = "WARN"
        else:
            gate.message = f"全部 {n_preprocess} clips进入预处理"

        self.gates.append(gate)

    def _check_score_threshold(self):
        """检查clip_score阈值过滤"""
        threshold = self.context.clip_score_threshold
        skip_enabled = self.context.skip_low_score_vlm

        # 统计分数分布
        scores = [c.clip_score for c in self.clip_results if c.clip_score > 0]
        skipped = [c for c in self.clip_results if c.filter_status == "SKIPPED"]
        passed = [c for c in self.clip_results if c.filter_status == "PASSED"]

        gate = RunGate(
            name="clip_score_threshold",
            passed=len(passed) > 0 or not skip_enabled,
            value=len(passed),
            threshold=f"clip_score >= {threshold}" if skip_enabled else "disabled",
            severity="FAIL" if (skip_enabled and len(passed) == 0 and len(skipped) > 0) else "INFO"
        )

        if not skip_enabled:
            gate.message = "阈值过滤已禁用"
        elif len(skipped) == 0:
            gate.message = f"无clip被阈值过滤 (threshold={threshold})"
        else:
            # 计算分布统计
            if scores:
                score_min = min(scores)
                score_max = max(scores)
                score_median = statistics.median(scores) if scores else 0
                score_mean = statistics.mean(scores) if scores else 0
            else:
                score_min = score_max = score_median = score_mean = 0

            gate.message = (
                f"阈值过滤: {len(skipped)}/{len(self.clip_results)} clips被跳过\n"
                f"  阈值: {threshold}\n"
                f"  分数分布: min={score_min:.3f}, median={score_median:.3f}, max={score_max:.3f}\n"
                f"  通过: {len(passed)}, 跳过: {len(skipped)}"
            )

            gate.details = {
                "threshold": threshold,
                "total": len(self.clip_results),
                "passed": len(passed),
                "skipped": len(skipped),
                "score_distribution": {
                    "min": score_min,
                    "max": score_max,
                    "median": score_median,
                    "mean": score_mean,
                }
            }

        self.gates.append(gate)

    def _check_vlm_input(self):
        """检查VLM输入数量"""
        n = self.context.n_vlm_analyzed

        gate = RunGate(
            name="vlm_input_clips",
            passed=n > 0,
            value=n,
            threshold=">0",
            severity="FAIL" if n == 0 else "INFO"
        )

        if n == 0:
            # 分析原因
            reasons = []
            if self.context.n_clips_cut == 0:
                reasons.append("无剪辑输出")
            elif self.context.n_preprocessed == 0:
                reasons.append("Top-K为0或无clip")
            else:
                # 检查是否全部被阈值过滤
                skipped = [c for c in self.clip_results if c.filter_status == "SKIPPED"]
                if len(skipped) == self.context.n_preprocessed:
                    reasons.append(f"全部 {len(skipped)} clips因clip_score < {self.context.clip_score_threshold}被跳过")

            gate.message = f"0 clips进入VLM! 原因: {'; '.join(reasons) if reasons else '未知'}"
        else:
            gate.message = f"{n} clips送入VLM分析"

        self.gates.append(gate)

    def _check_vlm_verdicts(self):
        """检查VLM verdict分布"""
        verdicts = [c.vlm_verdict for c in self.clip_results if c.vlm_verdict]
        if not verdicts:
            return

        from collections import Counter
        dist = Counter(verdicts)

        n_yes = dist.get("YES", 0)
        n_no = dist.get("NO", 0)
        n_post = dist.get("POST_EVENT_ONLY", 0)
        n_uncertain = dist.get("UNCERTAIN", 0)

        gate = RunGate(
            name="vlm_verdicts",
            passed=True,
            value=dict(dist),
            threshold="N/A",
            severity="INFO"
        )

        gate.message = f"VLM判决分布: YES={n_yes}, NO={n_no}, POST_EVENT_ONLY={n_post}, UNCERTAIN={n_uncertain}"
        self.gates.append(gate)

    def _compute_conclusion(self) -> str:
        """计算总体结论"""
        has_fail = any(g.severity == "FAIL" for g in self.gates)
        has_warn = any(g.severity == "WARN" for g in self.gates)

        if has_fail:
            return "FAIL"
        elif has_warn:
            return "WARN"
        else:
            return "PASS"

    def get_near_miss_clips(self, top_n: int = 10) -> List[Dict]:
        """获取最接近阈值的Top N clips（Near Miss分析）"""
        threshold = self.context.clip_score_threshold

        # 筛选被跳过的clips，按clip_score降序排列
        skipped = [
            c for c in self.clip_results
            if c.filter_status == "SKIPPED" and c.clip_score > 0
        ]
        skipped.sort(key=lambda x: x.clip_score, reverse=True)

        near_miss = []
        for c in skipped[:top_n]:
            gap = threshold - c.clip_score
            near_miss.append({
                "clip_id": c.clip_id,
                "clip_score": c.clip_score,
                "gap_to_threshold": gap,
                "threshold": threshold,
                "would_pass_at": c.clip_score - 0.001,  # 只需降低阈值到这个值即可通过
            })

        return near_miss

    def generate_recommendations(self) -> List[str]:
        """基于诊断结果生成建议"""
        recommendations = []

        # 检查是否有0 clips进入VLM的情况
        vlm_gate = next((g for g in self.gates if g.name == "vlm_input_clips"), None)
        if vlm_gate and not vlm_gate.passed:
            # 检查是否因为阈值过滤
            score_gate = next((g for g in self.gates if g.name == "clip_score_threshold"), None)
            if score_gate and hasattr(score_gate, 'details') and score_gate.details:
                details = score_gate.details
                if details.get("skipped", 0) > 0:
                    threshold = details["threshold"]
                    score_max = details["score_distribution"]["max"]

                    # 建议1：降低阈值
                    if score_max > 0:
                        suggested_threshold = max(0.15, score_max - 0.05)
                        recommendations.append(
                            f"建议临时降低clip_score_threshold从{threshold}到{suggested_threshold:.2f}，"
                            f"以便分析最高分的clips"
                        )

                    # 建议2：禁用阈值过滤
                    recommendations.append(
                        "或设置 skip_low_score_vlm=False 禁用阈值过滤，强制所有clips送VLM"
                    )

                    # 建议3：保留TopK兜底
                    recommendations.append(
                        "或实现'TopK兜底'逻辑：即使分数低于阈值，也保留Top3 clips送VLM"
                    )

        return recommendations


# ============ 报告生成器 ============

class ReportBuilder:
    """诊断报告生成器"""

    def __init__(
        self,
        context: RunContext,
        clip_results: List[ClipResult],
        output_dir: str = "reports",
        vlm_results: Optional[List[Dict]] = None,
    ):
        self.context = context
        self.clip_results = clip_results
        self.vlm_results = vlm_results or []

        # 确保run_id存在
        if not context.run_id:
            context.generate_run_id()

        self.output_dir = Path(output_dir) / context.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 诊断器
        self.diagnostics = ShortCircuitDiagnostics(context, clip_results)

    def build(self) -> str:
        """生成完整报告，返回报告目录路径"""
        try:
            # 1. 运行诊断
            conclusion, gates = self.diagnostics.run_diagnostics()
            near_miss = self.diagnostics.get_near_miss_clips()
            recommendations = self.diagnostics.generate_recommendations()

            # 2. 生成各文件
            self._write_env_txt()
            self._write_config_snapshot()
            self._write_stage_stats()
            self._write_clip_evidence()
            self._write_summary_json(conclusion, gates, near_miss, recommendations)
            self._write_report_json(conclusion, gates, near_miss, recommendations)
            self._write_report_md(conclusion, gates, near_miss, recommendations)

            print(f"\n[ReportBuilder] 诊断报告已生成: {self.output_dir}")
            return str(self.output_dir)

        except Exception as e:
            print(f"[ReportBuilder] 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _write_env_txt(self):
        """写入环境信息"""
        env_file = self.output_dir / "env.txt"
        lines = [
            f"Platform: {platform.platform()}",
            f"Python: {sys.version}",
            f"Working Dir: {os.getcwd()}",
            f"Run ID: {self.context.run_id}",
            f"Start Time: {self.context.start_time}",
            f"End Time: {self.context.end_time}",
        ]

        # GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                lines.append(f"CUDA: {torch.version.cuda}")
                lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
                lines.append(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except ImportError:
            lines.append("PyTorch: Not installed")

        with open(env_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _write_config_snapshot(self):
        """写入配置快照"""
        config_file = self.output_dir / "config_snapshot.yaml"

        config_data = self.context.config_snapshot
        if not config_data:
            config_data = {
                "clip_score_threshold": self.context.clip_score_threshold,
                "skip_low_score_vlm": self.context.skip_low_score_vlm,
                "top_clips": self.context.top_clips,
            }

        if HAS_YAML:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(safe_json_serialize(config_data), f, allow_unicode=True, default_flow_style=False)
        else:
            # fallback to JSON
            config_file = self.output_dir / "config_snapshot.json"
            with open(config_file, "w", encoding="utf-8") as f:
                safe_json_dump(config_data, f, ensure_ascii=False, indent=2)

    def _write_stage_stats(self):
        """写入阶段统计"""
        stats_file = self.output_dir / "stage_stats.json"

        stages_data = {}
        for name, stage in self.context.stages.items():
            stages_data[name] = stage.to_dict()

        # 确保关键阶段都有记录
        required_stages = [
            "embedding", "clip_sampler", "yolo", "preprocess_filter",
            "frame_sampling", "vlm", "ranking"
        ]
        for stage_name in required_stages:
            if stage_name not in stages_data:
                stages_data[stage_name] = {
                    "name": stage_name,
                    "duration_sec": 0.0,
                    "success": True,
                    "input_count": 0,
                    "output_count": 0,
                    "details": {"note": "未执行或未记录"}
                }

        with open(stats_file, "w", encoding="utf-8") as f:
            safe_json_dump(stages_data, f, ensure_ascii=False, indent=2)

    def _write_clip_evidence(self):
        """写入clip证据目录"""
        clips_dir = self.output_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        for clip in self.clip_results:
            clip_dir = clips_dir / clip.clip_id
            clip_dir.mkdir(exist_ok=True)

            # clip_meta.json
            meta = {
                "clip_id": clip.clip_id,
                "video_path": clip.video_path,
                "time_range": f"{clip.start_time:.1f}s - {clip.end_time:.1f}s",
                "duration": clip.duration,
                "clip_score": clip.clip_score,
                "accident_score": clip.accident_score,
                "final_score": clip.final_score,
                "filter_status": clip.filter_status,
                "skip_reason": clip.skip_reason,
                "thresholds_snapshot": {
                    "clip_score_threshold": self.context.clip_score_threshold,
                    "skip_low_score_vlm": self.context.skip_low_score_vlm,
                },
                "vlm_verdict": clip.vlm_verdict,
                "vlm_confidence": clip.vlm_confidence,
                "kept": clip.kept,
                "keep_reason": clip.keep_reason,
            }

            with open(clip_dir / "clip_meta.json", "w", encoding="utf-8") as f:
                safe_json_dump(meta, f, ensure_ascii=False, indent=2)

            # 对于SKIPPED的clip，生成占位说明
            if clip.filter_status == "SKIPPED":
                with open(clip_dir / "SKIPPED_README.txt", "w", encoding="utf-8") as f:
                    f.write(f"此clip因 {clip.skip_reason} 被跳过，未执行抽帧/VLM分析\n")
                    f.write(f"clip_score: {clip.clip_score}\n")
                    f.write(f"阈值: {self.context.clip_score_threshold}\n")

    def _write_summary_json(self, conclusion, gates, near_miss, recommendations):
        """写入简要摘要"""
        summary_file = self.output_dir / "summary.json"

        summary = {
            "run_id": self.context.run_id,
            "conclusion": conclusion,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.context.mode,
            "camera_id": self.context.camera_id,
            "total_duration_sec": self.context.get_total_duration(),
            "counts": {
                "n_source_videos": self.context.n_source_videos,
                "n_clips_cut": self.context.n_clips_cut,
                "n_preprocessed": self.context.n_preprocessed,
                "n_pass_score": self.context.n_pass_score,
                "n_vlm_analyzed": self.context.n_vlm_analyzed,
                "n_kept": self.context.n_kept,
            },
            "gates_summary": [
                {"name": g.name, "passed": g.passed, "severity": g.severity}
                for g in gates
            ],
            "has_near_miss": len(near_miss) > 0,
            "near_miss_count": len(near_miss),
            "recommendations_count": len(recommendations),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            safe_json_dump(summary, f, ensure_ascii=False, indent=2)

    def _write_report_json(self, conclusion, gates, near_miss, recommendations):
        """写入完整JSON报告"""
        report_file = self.output_dir / "report.json"

        report = {
            "run_id": self.context.run_id,
            "conclusion": conclusion,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": self.context.to_dict(),
            "gates": [g.to_dict() for g in gates],
            "near_miss_clips": near_miss,
            "recommendations": recommendations,
            "clips": [c.to_dict() for c in self.clip_results],
        }

        with open(report_file, "w", encoding="utf-8") as f:
            safe_json_dump(report, f, ensure_ascii=False, indent=2)

    def _write_report_md(self, conclusion, gates, near_miss, recommendations):
        """写入Markdown报告"""
        report_file = self.output_dir / "report.md"

        lines = []

        # 标题
        lines.append(f"# 事故分析诊断报告")
        lines.append("")
        lines.append(f"**Run ID**: `{self.context.run_id}`")
        lines.append(f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**模式**: {self.context.mode}")
        lines.append(f"**相机**: {self.context.camera_id}")
        lines.append("")

        # 总体结论
        conclusion_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(conclusion, "❓")
        lines.append(f"## {conclusion_emoji} 总体结论: **{conclusion}**")
        lines.append("")

        # 关键计数
        lines.append("### 关键计数")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 源视频数 | {self.context.n_source_videos} |")
        lines.append(f"| 剪辑clips | {self.context.n_clips_cut} |")
        lines.append(f"| 预处理clips | {self.context.n_preprocessed} |")
        lines.append(f"| 通过阈值 | {self.context.n_pass_score} |")
        lines.append(f"| VLM分析 | {self.context.n_vlm_analyzed} |")
        lines.append(f"| 最终保留 | {self.context.n_kept} |")
        lines.append("")

        # Run Gates（短路诊断）
        lines.append("## 🚦 Run Gates（短路诊断）")
        lines.append("")

        for gate in gates:
            emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "INFO": "ℹ️"}.get(
                "PASS" if gate.passed else gate.severity, "❓"
            )
            if gate.severity == "FAIL":
                lines.append(f"### {emoji} **[FAIL] {gate.name}**")
            elif gate.severity == "WARN":
                lines.append(f"### {emoji} **[WARN] {gate.name}**")
            else:
                lines.append(f"### {emoji} {gate.name}")

            lines.append("")
            lines.append(f"- **值**: `{gate.value}`")
            lines.append(f"- **阈值**: `{gate.threshold}`")

            # 多行message
            for msg_line in gate.message.split("\n"):
                lines.append(f"- {msg_line}")
            lines.append("")

        # Near Miss分析
        if near_miss:
            lines.append("## 🎯 Near Miss分析（最接近阈值的clips）")
            lines.append("")
            lines.append("以下clips的clip_score最接近阈值，稍微降低阈值即可通过：")
            lines.append("")
            lines.append("| Rank | Clip ID | clip_score | 距阈值 | 建议阈值 |")
            lines.append("|------|---------|------------|--------|----------|")

            for i, nm in enumerate(near_miss, 1):
                lines.append(
                    f"| {i} | `{nm['clip_id']}` | {nm['clip_score']:.3f} | "
                    f"{nm['gap_to_threshold']:.3f} | {nm['would_pass_at']:.3f} |"
                )
            lines.append("")

        # 建议
        if recommendations:
            lines.append("## 💡 建议")
            lines.append("")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Clip列表
        lines.append("## 📋 Clip列表")
        lines.append("")
        lines.append("| Clip ID | 时间范围 | clip_score | 状态 | 原因 | VLM |")
        lines.append("|---------|----------|------------|------|------|-----|")

        for c in self.clip_results:
            time_range = f"{c.start_time:.1f}-{c.end_time:.1f}s"
            status_emoji = {"PASSED": "✅", "SKIPPED": "⏭️", "NOT_SELECTED": "➖"}.get(c.filter_status, "❓")
            vlm_info = c.vlm_verdict if c.vlm_verdict else "-"

            lines.append(
                f"| `{c.clip_id}` | {time_range} | {c.clip_score:.3f} | "
                f"{status_emoji} {c.filter_status} | {c.skip_reason or '-'} | {vlm_info} |"
            )
        lines.append("")

        # 阶段统计
        lines.append("## ⏱️ 阶段统计")
        lines.append("")
        lines.append("| 阶段 | 耗时(s) | 输入 | 输出 | 跳过 | 状态 |")
        lines.append("|------|---------|------|------|------|------|")

        for name, stage in self.context.stages.items():
            status = "✅" if stage.success else "❌"
            lines.append(
                f"| {name} | {stage.duration_sec:.2f} | {stage.input_count} | "
                f"{stage.output_count} | {stage.skipped_count} | {status} |"
            )
        lines.append("")

        # 写入文件
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============ 便捷函数 ============

def create_report_from_pipeline(
    pipeline_result: Dict,
    config: Any,
    output_dir: str = "reports",
    run_id: Optional[str] = None,
) -> str:
    """从pipeline结果创建报告的便捷函数"""

    # 创建上下文
    context = RunContext()
    if run_id:
        context.run_id = run_id
    else:
        context.generate_run_id()

    # 从config提取阈值
    if hasattr(config, 'vlm'):
        context.clip_score_threshold = getattr(config.vlm, 'clip_score_threshold', 0.35)
        context.skip_low_score_vlm = getattr(config.vlm, 'skip_low_score_vlm', True)
        context.top_clips = getattr(config.vlm, 'top_clips', 3)

    # 从pipeline_result提取计数
    context.n_source_videos = pipeline_result.get("n_source_videos", 1)
    context.n_clips_cut = pipeline_result.get("n_clips_cut", 0)
    context.n_preprocessed = pipeline_result.get("n_preprocessed", 0)
    context.n_pass_score = pipeline_result.get("n_pass_score", 0)
    context.n_vlm_analyzed = pipeline_result.get("n_vlm_analyzed", 0)
    context.n_kept = pipeline_result.get("n_kept", 0)

    # 转换clip结果
    clip_results = []
    for clip_data in pipeline_result.get("clips", []):
        clip = ClipResult(
            clip_id=clip_data.get("clip_id", "unknown"),
            video_path=clip_data.get("video_path", ""),
            start_time=clip_data.get("start_time", 0),
            end_time=clip_data.get("end_time", 0),
            duration=clip_data.get("duration", 0),
            clip_score=clip_data.get("clip_score", 0),
            accident_score=clip_data.get("accident_score", 0),
            filter_status=clip_data.get("filter_status", "UNKNOWN"),
            skip_reason=clip_data.get("skip_reason", ""),
            vlm_verdict=clip_data.get("vlm_verdict", ""),
            kept=clip_data.get("kept", False),
        )
        clip_results.append(clip)

    # 生成报告
    builder = ReportBuilder(context, clip_results, output_dir)
    return builder.build()
