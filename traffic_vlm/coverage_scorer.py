"""
事故过程完整性评分模块 (Coverage Scorer)

用于评估候选clip是否覆盖【事故发生前→碰撞瞬间→事故后】的完整因果链。
包含完整过程的clip会获得更高的coverage_score，从而在排序中提升。

核心逻辑：
1. 估计碰撞时刻 t0（基于轨迹风险特征峰值）
2. 计算 coverage_score（过程完整性）
3. 计算 late_start_penalty（开始太晚的惩罚）
4. 计算 final_score 用于最终排序
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .config import CoverageConfig


def estimate_collision_time(
    clip: Dict,
    frame_results: Optional[List[Dict]] = None,
    tracks: Optional[Dict[int, Dict]] = None,
) -> Tuple[float, str]:
    """
    估计clip内的碰撞时刻 t0

    优先级：
    1. 最小距离峰值（两目标最近的时刻）
    2. 急刹/速度突变峰值
    3. 轨迹交叉时刻
    4. 回退到clip中点

    Args:
        clip: clip字典，包含 start_time, end_time, keyframes 等
        frame_results: 每帧检测结果 [{"timestamp": float, "detections": [...]}]
        tracks: 跟踪结果 {track_id: {"category": int, "trajectory": [(ts, x1, y1, x2, y2), ...]}}

    Returns:
        (t0, method): t0是相对于clip起始时间的碰撞时刻（秒），method是估计方法
    """
    clip_start = clip.get("start_time", 0.0)
    clip_end = clip.get("end_time", clip_start + 30.0)
    clip_duration = clip_end - clip_start

    # 默认回退：clip中点
    fallback_t0 = clip_duration / 2.0

    # 尝试从keyframes中提取风险特征时序
    keyframes = clip.get("keyframes", [])
    if not keyframes:
        return fallback_t0, "fallback_midpoint"

    # 方法1: 基于帧级风险分数找峰值
    risk_timeline = []
    for kf in keyframes:
        ts = kf.get("timestamp", 0.0) - clip_start  # 转为clip内相对时间
        if ts < 0:
            ts = 0

        # 提取风险指标
        collision_score = float(kf.get("collision_score", 0.0) or 0.0)
        track_intersection_score = float(kf.get("track_intersection_score", 0.0) or 0.0)
        deceleration_score = float(kf.get("deceleration_score", 0.0) or 0.0)
        accident_score = float(kf.get("accident_score", 0.0) or 0.0)

        # 综合风险分数（碰撞和急刹权重更高）
        combined_risk = (
            0.4 * collision_score +
            0.3 * deceleration_score +
            0.2 * track_intersection_score +
            0.1 * accident_score
        )

        risk_timeline.append((ts, combined_risk))

    if risk_timeline:
        # 找风险最高的时刻
        risk_timeline.sort(key=lambda x: x[1], reverse=True)
        peak_ts, peak_risk = risk_timeline[0]

        if peak_risk > 0.1:  # 有效风险信号
            return peak_ts, "risk_peak"

    # 方法2: 基于tracks计算最小距离时刻
    if tracks and len(tracks) >= 2:
        min_dist_ts = _find_min_distance_time(tracks, clip_start)
        if min_dist_ts is not None:
            return min_dist_ts, "min_distance"

    # 方法3: 基于frame_results检测IOU/距离峰值
    if frame_results:
        collision_ts = _find_collision_time_from_frames(frame_results, clip_start)
        if collision_ts is not None:
            return collision_ts, "frame_collision"

    return fallback_t0, "fallback_midpoint"


def _find_min_distance_time(tracks: Dict[int, Dict], clip_start: float) -> Optional[float]:
    """
    从轨迹数据中找到两目标最近的时刻

    Returns:
        相对于clip_start的时间（秒），如果找不到返回None
    """
    track_ids = list(tracks.keys())
    if len(track_ids) < 2:
        return None

    min_dist = float('inf')
    min_dist_ts = None

    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            traj1 = tracks[tid1].get("trajectory", [])
            traj2 = tracks[tid2].get("trajectory", [])

            if not traj1 or not traj2:
                continue

            # 对齐时间戳，找最近距离
            for t1 in traj1:
                if len(t1) < 5:
                    continue
                ts1, x1_1, y1_1, x2_1, y2_1 = t1[:5]
                c1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)

                for t2 in traj2:
                    if len(t2) < 5:
                        continue
                    ts2, x1_2, y1_2, x2_2, y2_2 = t2[:5]

                    # 时间戳接近才比较（±0.5秒）
                    if abs(ts1 - ts2) > 0.5:
                        continue

                    c2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

                    if dist < min_dist:
                        min_dist = dist
                        min_dist_ts = (ts1 + ts2) / 2 - clip_start

    if min_dist < 200 and min_dist_ts is not None:  # 距离阈值200像素
        return max(0, min_dist_ts)

    return None


def _find_collision_time_from_frames(frame_results: List[Dict], clip_start: float) -> Optional[float]:
    """
    从帧检测结果中找碰撞时刻（IOU/距离峰值）
    """
    max_collision_risk = 0.0
    collision_ts = None

    for frame in frame_results:
        ts = frame.get("timestamp", 0.0) - clip_start
        detections = frame.get("detections", [])

        if len(detections) < 2:
            continue

        # 两两计算距离
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                box_i = detections[i].get("bbox", [0, 0, 0, 0])
                box_j = detections[j].get("bbox", [0, 0, 0, 0])

                if len(box_i) < 4 or len(box_j) < 4:
                    continue

                # 计算IOU
                iou = _compute_iou(box_i, box_j)

                # 计算中心距离
                c1 = ((box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2)
                c2 = ((box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2)
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

                # 综合碰撞风险
                risk = iou + max(0, 1 - dist / 200)

                if risk > max_collision_risk:
                    max_collision_risk = risk
                    collision_ts = ts

    if max_collision_risk > 0.3 and collision_ts is not None:
        return max(0, collision_ts)

    return None


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个bbox的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_coverage_score(
    clip: Dict,
    collision_t0: float,
    config: "CoverageConfig",
) -> Dict[str, float]:
    """
    计算clip的过程完整性评分

    Args:
        clip: clip字典，包含 start_time, end_time
        collision_t0: 碰撞时刻（相对于clip起始时间，秒）
        config: CoverageConfig配置

    Returns:
        {
            "collision_t0": float,          # 碰撞时刻（clip内秒数）
            "pre_ok": float,                # 事故前覆盖度 (0~1)
            "post_ok": float,               # 事故后覆盖度 (0~1)
            "coverage_score": float,        # 综合覆盖度 (0~1)
            "late_start_penalty": float,    # 开始太晚惩罚 (>=0)
        }
    """
    clip_start = clip.get("start_time", 0.0)
    clip_end = clip.get("end_time", clip_start + 30.0)
    clip_duration = clip_end - clip_start

    # 计算事故前/后覆盖时长
    time_before_collision = collision_t0  # clip开始到碰撞的时间
    time_after_collision = clip_duration - collision_t0  # 碰撞到clip结束的时间

    # 覆盖度计算
    pre_ok = min(1.0, time_before_collision / config.pre_roll) if config.pre_roll > 0 else 1.0
    post_ok = min(1.0, time_after_collision / config.post_roll) if config.post_roll > 0 else 1.0

    # 综合覆盖度（乘法确保两边都需要覆盖）
    coverage_score = pre_ok * post_ok

    # 开始太晚惩罚（pre_roll时间内开始才不惩罚）
    late_start_penalty = max(0.0, config.pre_roll - time_before_collision)

    return {
        "collision_t0": collision_t0,
        "pre_ok": pre_ok,
        "post_ok": post_ok,
        "coverage_score": coverage_score,
        "late_start_penalty": late_start_penalty,
    }


def compute_final_score(
    base_score: float,
    coverage_result: Dict[str, float],
    config: "CoverageConfig",
) -> float:
    """
    计算最终排序分数

    final_score = base_score + lambda_coverage * coverage_score - mu_late * late_start_penalty

    Args:
        base_score: 原始clip_score
        coverage_result: compute_coverage_score的返回值
        config: CoverageConfig配置

    Returns:
        final_score
    """
    coverage_score = coverage_result.get("coverage_score", 0.0)
    late_start_penalty = coverage_result.get("late_start_penalty", 0.0)

    final_score = (
        base_score +
        config.lambda_coverage * coverage_score -
        config.mu_late * late_start_penalty
    )

    return final_score


def estimate_global_collision_time(clips: List[Dict]) -> Tuple[float, str, str]:
    """
    从所有clip中估计全局碰撞时刻（原视频时间）

    遍历所有clip的keyframes，找到风险最高的那个时刻作为全局t0。

    Args:
        clips: clip列表

    Returns:
        (global_t0, best_clip_id, method):
            global_t0是原视频时间（秒），
            best_clip_id是包含碰撞的clip ID，
            method是估计方法
    """
    best_risk = 0.0
    global_t0 = None
    best_clip_id = None

    for clip in clips:
        keyframes = clip.get("keyframes", [])

        for kf in keyframes:
            ts = kf.get("timestamp", 0.0)  # 原视频时间

            # 提取风险指标
            collision_score = float(kf.get("collision_score", 0.0) or 0.0)
            track_intersection_score = float(kf.get("track_intersection_score", 0.0) or 0.0)
            deceleration_score = float(kf.get("deceleration_score", 0.0) or 0.0)
            accident_score = float(kf.get("accident_score", 0.0) or 0.0)

            # 综合风险分数
            combined_risk = (
                0.4 * collision_score +
                0.3 * deceleration_score +
                0.2 * track_intersection_score +
                0.1 * accident_score
            )

            if combined_risk > best_risk:
                best_risk = combined_risk
                global_t0 = ts
                best_clip_id = clip.get("clip_id")

    if global_t0 is not None and best_risk > 0.1:
        return global_t0, best_clip_id, "global_risk_peak"

    # 回退：使用所有clip中最早的中点
    if clips:
        all_starts = [c.get("start_time", 0) for c in clips]
        all_ends = [c.get("end_time", 0) for c in clips]
        global_t0 = (min(all_starts) + min(all_ends)) / 2
        return global_t0, None, "fallback_earliest_midpoint"

    return 0.0, None, "fallback_zero"


def compute_coverage_score_global(
    clip: Dict,
    global_t0: float,
    config: "CoverageConfig",
) -> Dict[str, float]:
    """
    基于全局碰撞时刻计算clip的过程完整性评分

    Args:
        clip: clip字典，包含 start_time, end_time
        global_t0: 全局碰撞时刻（原视频时间，秒）
        config: CoverageConfig配置

    Returns:
        {
            "collision_t0": float,          # 碰撞时刻（原视频秒数）
            "collision_t0_relative": float, # 碰撞时刻相对于clip开始的秒数
            "pre_ok": float,                # 事故前覆盖度 (0~1)
            "post_ok": float,               # 事故后覆盖度 (0~1)
            "coverage_score": float,        # 综合覆盖度 (0~1)
            "late_start_penalty": float,    # 开始太晚惩罚 (>=0)
        }
    """
    clip_start = clip.get("start_time", 0.0)
    clip_end = clip.get("end_time", clip_start + 30.0)

    # 计算clip相对于全局碰撞时刻的覆盖
    # time_before_collision: clip开始时间到碰撞的时间（负数表示clip在碰撞后才开始）
    time_before_collision = global_t0 - clip_start
    # time_after_collision: 碰撞到clip结束的时间
    time_after_collision = clip_end - global_t0

    # 覆盖度计算
    # pre_ok: clip开始时间在碰撞前多久（越早越好，最多pre_roll秒）
    if time_before_collision >= config.pre_roll:
        pre_ok = 1.0
    elif time_before_collision > 0:
        pre_ok = time_before_collision / config.pre_roll
    else:
        # clip在碰撞后才开始，pre_ok = 0
        pre_ok = 0.0

    # post_ok: clip结束时间在碰撞后多久（越晚越好，最多post_roll秒）
    if time_after_collision >= config.post_roll:
        post_ok = 1.0
    elif time_after_collision > 0:
        post_ok = time_after_collision / config.post_roll
    else:
        # clip在碰撞前就结束了，post_ok = 0
        post_ok = 0.0

    # 综合覆盖度（乘法确保两边都需要覆盖）
    coverage_score = pre_ok * post_ok

    # 开始太晚惩罚
    # 如果clip开始时间在碰撞后，或者碰撞前覆盖不足pre_roll，就惩罚
    if time_before_collision < config.pre_roll:
        late_start_penalty = config.pre_roll - max(0, time_before_collision)
    else:
        late_start_penalty = 0.0

    return {
        "collision_t0": global_t0,
        "collision_t0_relative": time_before_collision,  # 可能为负
        "pre_ok": pre_ok,
        "post_ok": post_ok,
        "coverage_score": coverage_score,
        "late_start_penalty": late_start_penalty,
    }


def score_clips_with_coverage(
    clips: List[Dict],
    config: "CoverageConfig",
    frame_results_map: Optional[Dict[str, List[Dict]]] = None,
    tracks_map: Optional[Dict[str, Dict[int, Dict]]] = None,
) -> List[Dict]:
    """
    为所有clip计算coverage评分并重新排序

    核心逻辑：
    1. 首先从所有clip中估计一个全局碰撞时刻（原视频时间）
    2. 然后计算每个clip相对于全局碰撞时刻的覆盖度
    3. 包含"事故前+碰撞+事故后"完整过程的clip获得更高分

    Args:
        clips: clip列表
        config: CoverageConfig配置
        frame_results_map: {clip_id: [frame_results]} 可选
        tracks_map: {clip_id: tracks} 可选

    Returns:
        带coverage评分的clip列表（已按final_score降序排序）
    """
    if not config.enabled:
        return clips

    if not clips:
        return clips

    # 第一步：估计全局碰撞时刻
    global_t0, collision_clip_id, t0_method = estimate_global_collision_time(clips)

    scored_clips = []

    for clip in clips:
        clip_id = clip.get("clip_id", "")
        base_score = clip.get("clip_score", 0.0)

        # 基于全局碰撞时刻计算coverage评分
        coverage_result = compute_coverage_score_global(clip, global_t0, config)

        # 计算最终分数
        final_score = compute_final_score(base_score, coverage_result, config)

        # 添加评分字段到clip
        clip_with_score = clip.copy()
        clip_with_score.update({
            "collision_t0": coverage_result["collision_t0"],
            "collision_t0_relative": coverage_result["collision_t0_relative"],
            "collision_t0_method": t0_method,
            "collision_clip_id": collision_clip_id,  # 包含碰撞的clip
            "pre_ok": coverage_result["pre_ok"],
            "post_ok": coverage_result["post_ok"],
            "coverage_score": coverage_result["coverage_score"],
            "late_start_penalty": coverage_result["late_start_penalty"],
            "base_score": base_score,
            "final_score": final_score,
        })

        scored_clips.append(clip_with_score)

    # 按final_score降序排序
    scored_clips.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

    return scored_clips


def log_coverage_ranking(clips: List[Dict], logger=None):
    """
    输出coverage排名日志，便于调试

    Args:
        clips: 已评分的clip列表
        logger: 日志对象（可选，默认print）
    """
    log_fn = logger.info if logger else print

    # 获取全局碰撞时刻信息
    if clips:
        global_t0 = clips[0].get("collision_t0", 0)
        t0_method = clips[0].get("collision_t0_method", "unknown")
        collision_clip = clips[0].get("collision_clip_id", "N/A")
        log_fn(f"[CoverageScorer] 全局碰撞时刻: t0={global_t0:.1f}s (method={t0_method}, clip={collision_clip})")

    log_fn("[CoverageScorer] Clip排名（按final_score降序）:")
    log_fn("-" * 110)
    log_fn(f"{'Rank':>4} | {'Clip ID':<18} | {'Time Range':<16} | {'base':>6} | {'t0_rel':>7} | {'pre':>5} | {'post':>5} | {'cover':>6} | {'penalty':>7} | {'final':>7}")
    log_fn("-" * 110)

    for i, clip in enumerate(clips, 1):
        clip_id = clip.get("clip_id", "N/A")
        start = clip.get("start_time", 0)
        end = clip.get("end_time", 0)
        base = clip.get("base_score", clip.get("clip_score", 0))
        t0_rel = clip.get("collision_t0_relative", 0)  # 相对时间（可能为负）
        pre = clip.get("pre_ok", 0)
        post = clip.get("post_ok", 0)
        cover = clip.get("coverage_score", 0)
        penalty = clip.get("late_start_penalty", 0)
        final = clip.get("final_score", 0)

        log_fn(f"{i:>4} | {clip_id:<18} | {start:>6.1f}s-{end:>6.1f}s | {base:>6.4f} | {t0_rel:>7.1f} | {pre:>5.2f} | {post:>5.2f} | {cover:>6.4f} | {penalty:>7.2f} | {final:>7.4f}")

    log_fn("-" * 110)
