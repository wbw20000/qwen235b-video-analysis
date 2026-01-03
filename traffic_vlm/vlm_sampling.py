"""
VLM t0窗口高频抽帧模块

基于碰撞时刻t0进行高频采样，生成raw/roi/annotated三种输入图像。
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np

from .config import VLMSamplingConfig
from .roi_utils import (
    expand_box,
    union_box,
    clamp_to_image,
    crop_roi,
    create_event_window_roi,
    compute_union_roi_from_detections,
    find_risk_peak_center,
)


@dataclass
class SamplingResult:
    """采样结果"""
    timestamps: List[float]              # 采样时间戳列表
    raw_frames: List[Tuple[float, np.ndarray]]   # (ts, frame) 原图
    roi_frames: List[Tuple[float, np.ndarray]]   # (ts, roi) ROI图
    annotated_frames: List[Tuple[float, np.ndarray]]  # (ts, annotated) 标注图
    t0: float                            # 碰撞时刻
    t0_fallback: bool                    # 是否使用fallback
    t0_method: str                       # t0估计方法
    roi_mode: str                        # ROI模式
    roi_box: Optional[Tuple[int, int, int, int]]  # ROI边界框


def estimate_collision_time_from_clip(
    clip: Dict,
    frame_results: Optional[List[Dict]] = None,
    tracks: Optional[Dict] = None,
) -> Tuple[float, bool, str]:
    """
    估计clip内的碰撞时刻t0

    优先使用轨迹/风险特征，回退到clip中点。

    Args:
        clip: clip字典，包含 start_time, end_time, keyframes等
        frame_results: 帧检测结果列表
        tracks: 轨迹字典

    Returns:
        (t0, fallback, method):
            t0是clip内相对时间（秒）
            fallback是否为回退估计
            method是估计方法描述
    """
    clip_start = clip.get("start_time", 0.0)
    clip_end = clip.get("end_time", clip_start + 30.0)
    clip_duration = clip_end - clip_start

    # 方法1：从keyframes的风险分数找峰值
    keyframes = clip.get("keyframes", [])
    if keyframes:
        best_risk = 0.0
        best_ts = None

        for kf in keyframes:
            ts = kf.get("timestamp", 0.0)
            # 转换为clip内相对时间
            rel_ts = ts - clip_start if ts >= clip_start else ts

            # 综合风险分数
            collision = float(kf.get("collision_score", 0) or 0)
            decel = float(kf.get("deceleration_score", 0) or 0)
            intersect = float(kf.get("track_intersection_score", 0) or 0)
            accident = float(kf.get("accident_score", 0) or 0)

            risk = 0.4 * collision + 0.3 * decel + 0.2 * intersect + 0.1 * accident

            if risk > best_risk:
                best_risk = risk
                best_ts = rel_ts

        if best_ts is not None and best_risk > 0.1:
            # 确保t0在clip范围内
            t0 = max(0, min(best_ts, clip_duration))
            return (t0, False, "risk_peak")

    # 方法2：从frame_results找检测密度峰值
    if frame_results:
        max_detections = 0
        peak_idx = len(frame_results) // 2

        for i, fr in enumerate(frame_results):
            n_det = len(fr.get("detections", []))
            if n_det > max_detections:
                max_detections = n_det
                peak_idx = i

        if max_detections > 0:
            t0 = peak_idx * clip_duration / max(1, len(frame_results) - 1)
            return (t0, False, "detection_peak")

    # 回退：使用clip中点
    t0 = clip_duration / 2
    return (t0, True, "fallback_midpoint")


def compute_sampling_timestamps(
    t0: float,
    clip_duration: float,
    config: VLMSamplingConfig,
) -> List[float]:
    """
    计算采样时间戳列表

    Args:
        t0: 碰撞时刻（clip内相对时间）
        clip_duration: clip时长
        config: 采样配置

    Returns:
        采样时间戳列表（升序）
    """
    timestamps = set()

    # 1. t0窗口内的高频帧
    window_start = max(0, t0 - config.t0_window_pre)
    window_end = min(clip_duration, t0 + config.t0_window_post)
    window_duration = window_end - window_start

    n_window_frames = int(window_duration * config.fps) + 1
    for i in range(n_window_frames):
        ts = window_start + i / config.fps
        if 0 <= ts <= clip_duration:
            timestamps.add(round(ts, 3))

    # 2. 额外补帧：事故前
    extra_pre_ts = t0 - config.extra_pre
    if extra_pre_ts >= 0:
        timestamps.add(round(extra_pre_ts, 3))

    # 3. 额外补帧：事故后
    extra_post_ts = t0 + config.extra_post
    if extra_post_ts <= clip_duration:
        timestamps.add(round(extra_post_ts, 3))

    return sorted(timestamps)


def sample_frames_at_timestamps(
    video_path: str,
    timestamps: List[float],
    clip_start_offset: float = 0.0,
) -> List[Tuple[float, np.ndarray]]:
    """
    从视频中在指定时间戳采样帧

    Args:
        video_path: 视频路径
        timestamps: 时间戳列表（clip内相对时间）
        clip_start_offset: clip在原视频中的起始偏移

    Returns:
        [(timestamp, frame), ...] 列表
    """
    if not timestamps:
        return []

    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[vlm_sampling] 无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    for ts in timestamps:
        # 计算实际帧位置
        actual_ts = ts + clip_start_offset
        frame_idx = int(actual_ts * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret and frame is not None:
            frames.append((ts, frame))
        else:
            print(f"[vlm_sampling] 读取帧失败: ts={ts:.3f}s, frame_idx={frame_idx}")

    cap.release()
    return frames


def generate_roi_frames(
    raw_frames: List[Tuple[float, np.ndarray]],
    frame_results: Optional[List[Dict]],
    tracks: Optional[Dict],
    config: VLMSamplingConfig,
    track_ids: Optional[List[int]] = None,
) -> Tuple[List[Tuple[float, np.ndarray]], str, Optional[Tuple[int, int, int, int]]]:
    """
    从原始帧生成ROI帧

    Args:
        raw_frames: 原始帧列表
        frame_results: 检测结果
        tracks: 轨迹字典
        config: 采样配置
        track_ids: 指定的参与者ID

    Returns:
        (roi_frames, roi_mode, roi_box)
    """
    if not raw_frames:
        return [], config.roi_mode, None

    # 获取图像尺寸
    sample_frame = raw_frames[0][1]
    h, w = sample_frame.shape[:2]
    image_size = (w, h)

    roi_box = None
    roi_mode = config.roi_mode

    # 尝试union模式
    if config.roi_mode == "union" and frame_results:
        roi_box = compute_union_roi_from_detections(
            frame_results,
            scale=config.roi_scale,
            image_size=image_size,
            track_ids=track_ids,
        )
        if roi_box:
            roi_mode = "union"

    # 回退到event_window模式
    if roi_box is None:
        center = None
        if frame_results and tracks:
            center = find_risk_peak_center(frame_results, tracks)

        if center is None:
            center = (w // 2, h // 2)

        half = config.roi_window_size // 2
        x1, y1 = center[0] - half, center[1] - half
        x2, y2 = center[0] + half, center[1] + half
        roi_box = clamp_to_image((x1, y1, x2, y2), image_size)
        roi_mode = "event_window"

    # 生成ROI帧
    roi_frames = []
    for ts, frame in raw_frames:
        roi = crop_roi(frame, roi_box)
        roi_frames.append((ts, roi))

    return roi_frames, roi_mode, roi_box


def save_debug_frames(
    clip_id: str,
    raw_frames: List[Tuple[float, np.ndarray]],
    roi_frames: List[Tuple[float, np.ndarray]],
    annotated_frames: List[Tuple[float, np.ndarray]],
    debug_dir: str,
) -> Dict[str, List[str]]:
    """
    保存调试帧到文件

    Args:
        clip_id: clip ID
        raw_frames: 原始帧
        roi_frames: ROI帧
        annotated_frames: 标注帧
        debug_dir: 调试目录

    Returns:
        {"raw": [paths], "roi": [paths], "annotated": [paths]}
    """
    clip_debug_dir = os.path.join(debug_dir, clip_id)
    raw_dir = os.path.join(clip_debug_dir, "raw")
    roi_dir = os.path.join(clip_debug_dir, "roi")
    annotated_dir = os.path.join(clip_debug_dir, "annotated")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    paths = {"raw": [], "roi": [], "annotated": []}

    for ts, frame in raw_frames:
        path = os.path.join(raw_dir, f"frame_{ts:06.3f}.jpg")
        cv2.imwrite(path, frame)
        paths["raw"].append(path)

    for ts, roi in roi_frames:
        path = os.path.join(roi_dir, f"roi_{ts:06.3f}.jpg")
        cv2.imwrite(path, roi)
        paths["roi"].append(path)

    for ts, ann in annotated_frames:
        path = os.path.join(annotated_dir, f"annotated_{ts:06.3f}.jpg")
        cv2.imwrite(path, ann)
        paths["annotated"].append(path)

    return paths


def sample_clip_for_vlm(
    clip: Dict,
    video_path: str,
    config: VLMSamplingConfig,
    frame_results: Optional[List[Dict]] = None,
    tracks: Optional[Dict] = None,
    annotator_fn: Optional[callable] = None,
    track_ids: Optional[List[int]] = None,
) -> SamplingResult:
    """
    为VLM分析采样clip

    完整流程：估计t0 → 计算时间戳 → 抽帧 → 生成ROI → 生成标注

    Args:
        clip: clip字典
        video_path: 视频路径
        config: 采样配置
        frame_results: 检测结果（可选）
        tracks: 轨迹字典（可选）
        annotator_fn: 标注函数（可选）
        track_ids: 参与者ID列表（可选）

    Returns:
        SamplingResult
    """
    clip_id = clip.get("clip_id", "unknown")
    clip_start = clip.get("start_time", 0.0)
    clip_end = clip.get("end_time", clip_start + 30.0)
    clip_duration = clip_end - clip_start

    # 判断是否使用剪辑后的视频
    clip_source = clip.get("clip_source", "original")
    if clip_source == "clipped":
        actual_video_path = clip.get("video_path", video_path)
        clip_start_offset = 0.0  # 剪辑后的视频从0开始
    else:
        actual_video_path = video_path
        clip_start_offset = clip_start

    # 1. 估计碰撞时刻
    t0, t0_fallback, t0_method = estimate_collision_time_from_clip(
        clip, frame_results, tracks
    )
    print(f"[vlm_sampling] {clip_id}: t0={t0:.2f}s ({t0_method}), fallback={t0_fallback}")

    # 2. 计算采样时间戳
    timestamps = compute_sampling_timestamps(t0, clip_duration, config)
    print(f"[vlm_sampling] {clip_id}: 采样{len(timestamps)}帧, 时间戳={timestamps[:5]}...")

    # 3. 采样原始帧
    raw_frames = sample_frames_at_timestamps(
        actual_video_path, timestamps, clip_start_offset
    )

    if not raw_frames:
        print(f"[vlm_sampling] {clip_id}: 采样失败，无帧返回")
        return SamplingResult(
            timestamps=[],
            raw_frames=[],
            roi_frames=[],
            annotated_frames=[],
            t0=t0,
            t0_fallback=t0_fallback,
            t0_method=t0_method,
            roi_mode="none",
            roi_box=None,
        )

    # 4. 生成ROI帧
    roi_frames, roi_mode, roi_box = generate_roi_frames(
        raw_frames, frame_results, tracks, config, track_ids
    )

    # 5. 生成标注帧（如果有标注函数）
    annotated_frames = []
    if annotator_fn:
        try:
            annotated_frames = annotator_fn(raw_frames, frame_results, tracks)
        except Exception as e:
            print(f"[vlm_sampling] 标注失败: {e}")
            annotated_frames = raw_frames  # 回退到原图

    # 6. 保存调试帧
    if config.debug_dump:
        debug_paths = save_debug_frames(
            clip_id, raw_frames, roi_frames, annotated_frames, config.debug_dir
        )
        print(f"[vlm_sampling] 调试帧已保存: {config.debug_dir}/{clip_id}/")

    return SamplingResult(
        timestamps=timestamps,
        raw_frames=raw_frames,
        roi_frames=roi_frames,
        annotated_frames=annotated_frames if annotated_frames else raw_frames,
        t0=t0,
        t0_fallback=t0_fallback,
        t0_method=t0_method,
        roi_mode=roi_mode,
        roi_box=roi_box,
    )


def format_sampling_log(clip_id: str, result: SamplingResult) -> str:
    """格式化采样日志"""
    return (
        f"[vlm_sampling] {clip_id}: "
        f"t0={result.t0:.2f}s ({result.t0_method}), "
        f"fallback={result.t0_fallback}, "
        f"n_frames={len(result.timestamps)}, "
        f"roi_mode={result.roi_mode}, "
        f"roi_box={result.roi_box}"
    )
