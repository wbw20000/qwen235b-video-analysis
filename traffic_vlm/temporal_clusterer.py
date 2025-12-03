from __future__ import annotations

from typing import Dict, List
import uuid

from .config import ClusterConfig


def cluster_frames_to_clips(frame_candidates: List[Dict], cluster_config: ClusterConfig) -> List[Dict]:
    """
    帧级候选 -> 时间聚类 -> clip 列表。
    frame_candidates: [{"metadata": {...}, "similarity_score": float, ...}]
    metadata 必须包含 timestamp, camera_id, image_path。
    """
    if not frame_candidates:
        return []

    # 按时间排序
    sorted_frames = sorted(frame_candidates, key=lambda x: x["metadata"].get("timestamp", 0.0))
    clusters: List[Dict] = []

    for item in sorted_frames:
        ts = item["metadata"].get("timestamp", 0.0)
        camera_id = item["metadata"].get("camera_id", "cam")
        if not clusters:
            clusters.append(
                {
                    "camera_id": camera_id,
                    "timestamps": [ts],
                    "frames": [item],
                }
            )
            continue

        last_cluster = clusters[-1]
        last_ts = last_cluster["timestamps"][-1]
        if ts - last_ts <= cluster_config.merge_window_seconds:
            last_cluster["timestamps"].append(ts)
            last_cluster["frames"].append(item)
        else:
            clusters.append(
                {
                    "camera_id": camera_id,
                    "timestamps": [ts],
                    "frames": [item],
                }
            )

    clips: List[Dict] = []
    for cluster in clusters:
        ts_list = cluster["timestamps"]
        start_ts = min(ts_list) - cluster_config.pre_padding
        end_ts = max(ts_list) + cluster_config.post_padding
        clip_score = max(f["similarity_score"] for f in cluster["frames"])
        clip_id = f"clip-{uuid.uuid4().hex[:8]}"
        clips.append(
            {
                "clip_id": clip_id,
                "camera_id": cluster["camera_id"],
                "start_time": max(0.0, start_ts),
                "end_time": end_ts,
                "clip_score": clip_score,
                "keyframes": [f["metadata"] for f in cluster["frames"]],
            }
        )

    clips = sorted(clips, key=lambda x: x["clip_score"], reverse=True)
    return clips[: cluster_config.candidate_clip_top_k]
