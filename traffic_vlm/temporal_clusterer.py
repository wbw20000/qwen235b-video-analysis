from __future__ import annotations

from typing import Dict, List
import uuid

from .config import ClusterConfig


def _frame_accident_score(frame: Dict) -> float:
    """
    统一提取事故相关信号。后续可以在 metadata 中补充：
    - accident_score: 直接的事故置信度
    - collision_score / track_intersection_score / deceleration_score: 轨迹/碰撞信号
    - accident_template_hit: 命中事故模板的标记
    """
    meta = frame.get("metadata", {}) if frame else {}
    scores = [
        float(meta.get("accident_score", 0.0) or 0.0),
        float(meta.get("collision_score", 0.0) or 0.0),
        float(meta.get("track_intersection_score", 0.0) or 0.0),
        float(meta.get("deceleration_score", 0.0) or 0.0),
    ]
    if meta.get("accident_template_hit"):
        scores.append(1.0)
    return max(scores) if scores else 0.0


def _cluster_center_ts(cluster: Dict) -> float:
    ts_list = cluster.get("timestamps", [])
    if not ts_list:
        return 0.0
    return sum(ts_list) / len(ts_list)


def cluster_frames_to_clips(frame_candidates: List[Dict], cluster_config: ClusterConfig, accident_mode: bool = False) -> List[Dict]:
    """
    帧级候选 -> 时间聚类 -> clip 列表。
    frame_candidates: [{"metadata": {...}, "similarity_score": float, ...}]
    metadata 必须包含 timestamp, camera_id, image_path。

    新增：
    - 限制单个 cluster 的最大时长，防止 clip 覆盖整个视频。
    - 基于 cluster 中心时间的窗口聚合，可在事故模式下放宽窗口/时长。
    - 事故信号可放宽时间间隔，并参与 clip_score 计算。
    """
    if not frame_candidates:
        return []

    # 按时间排序
    sorted_frames = sorted(frame_candidates, key=lambda x: x["metadata"].get("timestamp", 0.0))
    clusters: List[Dict] = []

    # 获取最大clip时长配置（默认30秒）
    default_max_clip_duration = getattr(cluster_config, "max_clip_duration", 30.0)
    accident_max_clip_duration = getattr(cluster_config, "accident_max_clip_duration", default_max_clip_duration)
    default_merge_window = getattr(cluster_config, "merge_window_seconds", 5.0)
    accident_merge_window = getattr(cluster_config, "accident_merge_window_seconds", default_merge_window)
    accident_threshold = getattr(cluster_config, "accident_score_threshold", 0.35)

    for item in sorted_frames:
        ts = item["metadata"].get("timestamp", 0.0)
        camera_id = item["metadata"].get("camera_id", "cam")
        frame_acc_score = _frame_accident_score(item)
        if not clusters:
            clusters.append(
                {
                    "camera_id": camera_id,
                    "timestamps": [ts],
                    "frames": [item],
                    "accident_score": frame_acc_score,
                }
            )
            continue

        last_cluster = clusters[-1]
        first_ts = last_cluster["timestamps"][0]

        # 根据事故信号动态调整时间窗口与最大时长
        cluster_acc_score = last_cluster.get("accident_score", 0.0)
        use_accident_window = accident_mode or frame_acc_score >= accident_threshold or cluster_acc_score >= accident_threshold
        merge_window = accident_merge_window if use_accident_window else default_merge_window
        max_clip_duration = accident_max_clip_duration if use_accident_window else default_max_clip_duration

        cluster_duration = ts - first_ts
        cluster_center = _cluster_center_ts(last_cluster)
        within_center_window = abs(ts - cluster_center) <= merge_window
        within_max_duration = cluster_duration <= max_clip_duration

        # 事故信号可放宽窗口，只要不超过最大时长
        if (within_center_window and within_max_duration) or (use_accident_window and within_max_duration):
            last_cluster["timestamps"].append(ts)
            last_cluster["frames"].append(item)
            last_cluster["accident_score"] = max(cluster_acc_score, frame_acc_score)
        else:
            # 开始新的 cluster
            clusters.append(
                {
                    "camera_id": camera_id,
                    "timestamps": [ts],
                    "frames": [item],
                    "accident_score": frame_acc_score,
                }
            )

    clips: List[Dict] = []
    for cluster in clusters:
        ts_list = cluster["timestamps"]
        cluster_acc_score = cluster.get("accident_score", 0.0)
        is_accident_cluster = cluster_acc_score >= accident_threshold

        # 根据事故簇使用更长的前后缓冲
        padding_pre = cluster_config.accident_pre_padding if is_accident_cluster else cluster_config.pre_padding
        padding_post = cluster_config.accident_post_padding if is_accident_cluster else cluster_config.post_padding

        start_ts = min(ts_list) - padding_pre
        end_ts = max(ts_list) + padding_post
        max_similarity = max(f["similarity_score"] for f in cluster["frames"])
        clip_score = (1.0 - cluster_config.accident_score_weight) * max_similarity + cluster_config.accident_score_weight * cluster_acc_score
        # clip_score 至少保持相似度的下限
        clip_score = max(clip_score, max_similarity)

        # 长版剪辑的时间范围（仅事故簇）
        long_version = None
        if is_accident_cluster:
            long_pre = padding_pre + cluster_config.accident_long_extra_pre
            long_post = padding_post + cluster_config.accident_long_extra_post
            long_version = {
                "start_time": max(0.0, min(ts_list) - long_pre),
                "end_time": max(ts_list) + long_post,
            }

        clip_id = f"clip-{uuid.uuid4().hex[:8]}"
        clips.append(
            {
                "clip_id": clip_id,
                "camera_id": cluster["camera_id"],
                "start_time": max(0.0, start_ts),
                "end_time": end_ts,
                "clip_score": clip_score,
                "keyframes": [f["metadata"] for f in cluster["frames"]],
                "accident_score": cluster_acc_score,
                "is_accident": is_accident_cluster,
                "long_version": long_version,
            }
        )

    clips = sorted(clips, key=lambda x: x["clip_score"], reverse=True)
    return clips[: cluster_config.candidate_clip_top_k]
