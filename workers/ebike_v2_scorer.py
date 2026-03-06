#!/usr/bin/env python3
"""
Ebike Violation V2 Scorer - margin + Top-K 打分策略

特性：
1. margin = pos - 0.7*hn - 0.3*neg（三元组评分）
2. Top-K帧选择代替固定阈值
3. Feature Flag控制（EBIKE_V2_SCORING）
4. 不影响accident pipeline

使用方式：
    from workers.ebike_v2_scorer import EbikeV2Scorer, EBIKE_V2_ENABLED

    if EBIKE_V2_ENABLED and analysis_type == "ebike_violation":
        scorer = EbikeV2Scorer(embedding_fn)
        frames_with_margin = scorer.compute_margin_scores(frames)
        candidate_frames = scorer.select_topk_frames(frames_with_margin)
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    import yaml
except ImportError:
    yaml = None

log = logging.getLogger(__name__)

# ==================== Feature Flag ====================
EBIKE_V2_ENABLED = os.getenv("EBIKE_V2_SCORING", "false").lower() == "true"

# ==================== Gate 配置 ====================
# Top-M Clip Gate: 只选择 Top M 个 clips 送 VLM
def _parse_top_m() -> int:
    """解析 EBIKE_V2_TOP_M，合法范围 1-5，非法值回退默认 2"""
    raw = os.getenv("EBIKE_V2_TOP_M", "2")
    try:
        val = int(raw)
        if 1 <= val <= 5:
            return val
    except (ValueError, TypeError):
        pass
    return 2

EBIKE_V2_TOP_M = _parse_top_m()

# Camera Cooldown Gate: 同一 camera_id 在冷却期内最多触发 1 次 ebike VLM
def _parse_cooldown_sec() -> int:
    """解析 EBIKE_V2_COOLDOWN_SEC，合法范围 0-600，0 表示关闭"""
    raw = os.getenv("EBIKE_V2_COOLDOWN_SEC", "60")
    try:
        val = int(raw)
        if 0 <= val <= 600:
            return val
    except (ValueError, TypeError):
        pass
    return 60

EBIKE_V2_COOLDOWN_SEC = _parse_cooldown_sec()


@dataclass
class EbikeGroupEmbeddings:
    """Ebike模板组嵌入"""
    name: str
    weight: float
    positives: np.ndarray      # [N_pos, dim]
    hard_negatives: np.ndarray  # [N_hn, dim]
    negatives: np.ndarray       # [N_neg, dim]


@dataclass
class EbikeFrameScore:
    """Ebike帧分数"""
    frame_idx: int
    timestamp_sec: float
    margin: float                 # margin = pos - 0.7*hn - 0.3*neg
    pos_sim: float                # 最高正样本相似度
    hn_sim: float                 # 最高hard_negative相似度
    neg_sim: float                # 最高negative相似度
    best_group: str               # 最佳匹配组
    group_margins: Dict[str, float] = field(default_factory=dict)


@dataclass
class EbikeClusterConfig:
    """Ebike聚类配置（独立于accident）"""
    min_gap_sec: float = 3.0
    min_clip_frames: int = 2
    pre_roll_sec: float = 2.0
    post_roll_sec: float = 2.0
    top_k_frames: int = 30


class EbikeV2Scorer:
    """
    Ebike V2 打分器

    实现 margin + Top-K 策略，仅对 ebike_violation 生效
    """

    # 默认评分参数
    DEFAULT_BETA = 0.7   # hard_negative惩罚系数
    DEFAULT_GAMMA = 0.3  # negative惩罚系数
    DEFAULT_TOP_K = 30   # Top-K帧数

    def __init__(
        self,
        embedding_fn: callable,
        template_path: Optional[str] = None,
        beta: float = None,
        gamma: float = None,
        top_k: int = None
    ):
        """
        初始化V2打分器

        Args:
            embedding_fn: 文本嵌入函数，接收List[str]返回np.ndarray
            template_path: 模板文件路径，默认使用ebike_violation_v2.yaml
            beta: hard_negative惩罚系数
            gamma: negative惩罚系数
            top_k: Top-K帧数
        """
        self.embedding_fn = embedding_fn

        if template_path is None:
            base_dir = Path(__file__).parent.parent
            template_path = str(base_dir / "traffic_vlm/templates/ebike_violation_v2.yaml")

        self.template_path = template_path
        self.beta = beta
        self.gamma = gamma
        self.top_k = top_k

        # 延迟加载
        self._groups: List[EbikeGroupEmbeddings] = []
        self._config: EbikeClusterConfig = EbikeClusterConfig()
        self._loaded = False

    def _load_template(self):
        """加载模板并计算嵌入"""
        if self._loaded:
            return

        if yaml is None:
            log.error("PyYAML not installed")
            return

        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 解析评分参数
            scoring = data.get("scoring", {})
            if self.beta is None:
                self.beta = scoring.get("beta_hardneg", self.DEFAULT_BETA)
            if self.gamma is None:
                self.gamma = scoring.get("gamma_neg", self.DEFAULT_GAMMA)
            if self.top_k is None:
                self.top_k = scoring.get("top_k_frames", self.DEFAULT_TOP_K)

            # 解析聚类参数
            clustering = data.get("clustering", {})
            self._config = EbikeClusterConfig(
                min_gap_sec=clustering.get("min_gap_sec", 3.0),
                min_clip_frames=clustering.get("min_clip_frames", 2),
                pre_roll_sec=clustering.get("pre_roll_sec", 2.0),
                post_roll_sec=clustering.get("post_roll_sec", 2.0),
                top_k_frames=self.top_k
            )

            # 计算组嵌入
            for group_data in data.get("groups", []):
                name = group_data.get("name", "unnamed")
                weight = group_data.get("weight", 1.0)

                positives = group_data.get("positives", [])
                hard_negatives = group_data.get("hard_negatives", [])
                negatives = group_data.get("negatives", [])

                # 计算嵌入
                pos_emb = self._compute_embeddings(positives)
                hn_emb = self._compute_embeddings(hard_negatives)
                neg_emb = self._compute_embeddings(negatives)

                self._groups.append(EbikeGroupEmbeddings(
                    name=name,
                    weight=weight,
                    positives=pos_emb,
                    hard_negatives=hn_emb,
                    negatives=neg_emb
                ))

            self._loaded = True
            log.info(f"EbikeV2Scorer loaded: {len(self._groups)} groups, "
                    f"beta={self.beta}, gamma={self.gamma}, top_k={self.top_k}")

        except Exception as e:
            log.error(f"Failed to load ebike template: {e}")

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """计算文本嵌入"""
        if not texts:
            return np.zeros((0, 768))

        try:
            embeddings = self.embedding_fn(texts)
            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            return embeddings / norms
        except Exception as e:
            log.error(f"Embedding failed: {e}")
            return np.zeros((0, 768))

    def _max_cosine_sim(self, query: np.ndarray, embeddings: np.ndarray) -> float:
        """计算最大余弦相似度"""
        if embeddings is None or len(embeddings) == 0:
            return 0.0

        # query已归一化，embeddings已归一化
        similarities = np.dot(embeddings, query)
        return float(np.max(similarities))

    def compute_margin_scores(
        self,
        frames: List[Any],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[EbikeFrameScore]:
        """
        计算每帧的margin分数

        公式：margin = pos - beta*hn - gamma*neg

        Args:
            frames: 帧列表，每帧需要有 embedding 和 timestamp_sec 属性

        Returns:
            EbikeFrameScore列表
        """
        self._load_template()

        if not self._groups:
            log.warning("No ebike groups loaded")
            return []

        scores = []

        for i, frame in enumerate(frames):
            if not hasattr(frame, 'embedding') or frame.embedding is None:
                continue

            # 归一化帧嵌入
            frame_emb = np.array(frame.embedding).flatten()
            frame_norm = frame_emb / (np.linalg.norm(frame_emb) + 1e-8)

            group_margins = {}
            best_margin = -float('inf')
            best_group = ""
            best_pos = 0.0
            best_hn = 0.0
            best_neg = 0.0

            for group in self._groups:
                pos_sim = self._max_cosine_sim(frame_norm, group.positives)
                hn_sim = self._max_cosine_sim(frame_norm, group.hard_negatives)
                neg_sim = self._max_cosine_sim(frame_norm, group.negatives)

                # margin = pos - beta*hn - gamma*neg
                margin = pos_sim - self.beta * hn_sim - self.gamma * neg_sim
                weighted_margin = margin * group.weight

                group_margins[group.name] = weighted_margin

                if weighted_margin > best_margin:
                    best_margin = weighted_margin
                    best_group = group.name
                    best_pos = pos_sim
                    best_hn = hn_sim
                    best_neg = neg_sim

            timestamp = getattr(frame, 'timestamp_sec', i)
            frame_idx = getattr(frame, 'frame_idx', i)

            scores.append(EbikeFrameScore(
                frame_idx=frame_idx,
                timestamp_sec=timestamp,
                margin=best_margin,
                pos_sim=best_pos,
                hn_sim=best_hn,
                neg_sim=best_neg,
                best_group=best_group,
                group_margins=group_margins
            ))

        return scores

    def select_topk_frames(
        self,
        frame_scores: List[EbikeFrameScore],
        top_k: int = None
    ) -> List[EbikeFrameScore]:
        """
        选择Top-K高margin帧

        Args:
            frame_scores: 帧分数列表
            top_k: Top-K数量，默认使用配置值

        Returns:
            Top-K帧分数列表（按margin降序）
        """
        if top_k is None:
            top_k = self.top_k or self.DEFAULT_TOP_K

        # 按margin降序排序
        sorted_scores = sorted(frame_scores, key=lambda x: x.margin, reverse=True)
        return sorted_scores[:top_k]

    def cluster_frames_to_clips(
        self,
        frame_scores: List[EbikeFrameScore],
        original_frames: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        将Top-K帧聚类为视频片段

        使用ebike专用聚类参数，不影响accident

        Args:
            frame_scores: 已排序的帧分数列表
            original_frames: 原始帧对象列表（用于获取帧路径等信息）

        Returns:
            片段列表
        """
        if not frame_scores:
            return []

        # 按时间排序
        sorted_by_time = sorted(frame_scores, key=lambda x: x.timestamp_sec)

        clips = []
        current_clip = [sorted_by_time[0]]

        for score in sorted_by_time[1:]:
            gap = score.timestamp_sec - current_clip[-1].timestamp_sec

            if gap <= self._config.min_gap_sec:
                current_clip.append(score)
            else:
                # 保存当前片段
                if len(current_clip) >= self._config.min_clip_frames:
                    clips.append(self._build_clip(current_clip, len(clips), original_frames))
                current_clip = [score]

        # 处理最后一个片段
        if len(current_clip) >= self._config.min_clip_frames:
            clips.append(self._build_clip(current_clip, len(clips), original_frames))

        return clips

    def _build_clip(
        self,
        scores: List[EbikeFrameScore],
        clip_id: int,
        original_frames: List[Any]
    ) -> Dict[str, Any]:
        """构建片段信息"""
        margins = [s.margin for s in scores]

        # 获取原始帧对象
        frame_indices = [s.frame_idx for s in scores]
        clip_frames = []
        for idx in frame_indices:
            if idx < len(original_frames):
                clip_frames.append(original_frames[idx])

        return {
            "clip_id": clip_id,
            "start_sec": scores[0].timestamp_sec - self._config.pre_roll_sec,
            "end_sec": scores[-1].timestamp_sec + self._config.post_roll_sec,
            "frame_count": len(scores),
            "clip_score": max(margins),          # 最大margin
            "mean_margin": float(np.mean(margins)),
            "best_group": scores[0].best_group,  # 取最高分帧的组
            "frame_scores": scores,
            "frames": clip_frames,
            "group_distribution": self._count_groups(scores)
        }

    def _count_groups(self, scores: List[EbikeFrameScore]) -> Dict[str, int]:
        """统计各组命中次数"""
        counts = {}
        for s in scores:
            counts[s.best_group] = counts.get(s.best_group, 0) + 1
        return counts

    def get_stats(self, frame_scores: List[EbikeFrameScore]) -> Dict[str, Any]:
        """
        获取统计信息（用于日志）

        Args:
            frame_scores: 帧分数列表

        Returns:
            统计字典
        """
        if not frame_scores:
            return {"count": 0}

        margins = [s.margin for s in frame_scores]

        # 按组统计
        group_counts = {}
        for s in frame_scores:
            group_counts[s.best_group] = group_counts.get(s.best_group, 0) + 1

        return {
            "count": len(frame_scores),
            "margin_max": float(max(margins)),
            "margin_mean": float(np.mean(margins)),
            "margin_p95": float(np.percentile(margins, 95)) if len(margins) >= 20 else float(max(margins)),
            "margin_min": float(min(margins)),
            "group_distribution": group_counts
        }


def get_ebike_v2_scorer(embedding_fn: callable) -> Optional[EbikeV2Scorer]:
    """
    获取EbikeV2Scorer实例（仅当Feature Flag开启时）

    Args:
        embedding_fn: 嵌入函数

    Returns:
        EbikeV2Scorer实例，如果Feature Flag关闭则返回None
    """
    if not EBIKE_V2_ENABLED:
        return None

    return EbikeV2Scorer(embedding_fn)


# ==================== 日志结构 ====================

def log_ebike_v2_result(
    logger,
    job_id: str,
    trace_id: str,
    camera_id: str,
    stats: Dict[str, Any],
    clips: List[Dict],
    vlm_calls: int,
    vlm_time_ms: float
):
    """
    输出ebike V2结构化日志

    格式：JSON单行，便于日志聚合分析
    """
    log_data = {
        "event": "ebike_v2_analysis",
        "trace_id": trace_id,
        "job_id": job_id,
        "camera_id": camera_id,
        "margin_stats": {
            "max": stats.get("margin_max", 0),
            "mean": stats.get("margin_mean", 0),
            "p95": stats.get("margin_p95", 0)
        },
        "topk_by_group": stats.get("group_distribution", {}),
        "clips_count": len(clips),
        "vlm_calls": vlm_calls,
        "vlm_time_ms": vlm_time_ms
    }

    logger.info(f"EBIKE_V2: {json.dumps(log_data, ensure_ascii=False)}")
