"""
文件级事件定位模块 (File Event Locator)

核心功能：
1. 以单个mp4文件为样本，生成 risk_timeline
2. 在文件内找 TopN 风险峰值时刻 t_event
3. 生成 file-level 候选或 event-window 子候选
4. 计算 t0_validity 验证碰撞时刻的可靠性

用于解决：
- clip-a826cb3b（完整事故）排名不靠前的问题
- clip-372493c8（无事故）因coverage虚高排第一的问题
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import uuid


@dataclass
class FileEventConfig:
    """文件级事件定位配置"""

    # 风险时间序列采样
    risk_sampling_fps: float = 2.0       # 采样频率(Hz)，1~2Hz低成本扫描

    # 峰值检测
    top_n_peaks: int = 5                  # TopN风险峰值数量
    peak_min_distance_sec: float = 3.0    # 峰值间最小间距(秒)
    peak_threshold: float = 0.15          # 峰值阈值(0~1)

    # 子候选窗口
    pre_roll: float = 8.0                 # 事故前覆盖时长(秒)
    post_roll: float = 12.0               # 事故后覆盖时长(秒)
    merge_gap_sec: float = 3.0            # 子候选合并间隔(秒)

    # 候选模式
    candidate_mode: str = "file_plus_subclips"  # file_only | file_plus_subclips

    # t0_validity 验证阈值
    min_distance_threshold: float = 150.0    # 最小距离阈值(像素)
    distance_drop_threshold: float = 0.3     # 距离突降阈值(相对值)
    velocity_change_threshold: float = 0.4   # 速度变化阈值(相对值)
    iou_contact_threshold: float = 0.05      # IoU接触阈值

    # 有条件保留策略阈值
    validity_threshold: float = 0.3          # t0_validity保留阈值
    risk_peak_threshold: float = 0.25        # risk_peak保留阈值
    roi_median_threshold: float = 80.0       # ROI中值边长阈值(像素)


@dataclass
class RiskTimelineResult:
    """风险时间序列分析结果"""

    timestamps: List[float] = field(default_factory=list)  # 时间点列表
    risk_scores: List[float] = field(default_factory=list)  # 对应风险分数
    peak_times: List[float] = field(default_factory=list)   # 峰值时间点
    peak_scores: List[float] = field(default_factory=list)  # 峰值分数
    max_risk: float = 0.0                                   # 最大风险分数
    max_risk_time: float = 0.0                              # 最大风险时间点


@dataclass
class T0ValidityResult:
    """t0有效性验证结果"""

    t0: float = 0.0                           # 碰撞时刻估计
    t0_fallback: bool = True                  # 是否回退估计
    t0_method: str = "fallback_midpoint"      # 估计方法
    validity: float = 0.0                     # 有效性分数(0~1)
    validity_reason: str = ""                 # 有效性判断依据

    # 各项证据
    min_distance_evidence: float = 0.0        # 最小距离证据
    distance_drop_evidence: float = 0.0       # 距离突降证据
    velocity_change_evidence: float = 0.0     # 速度变化证据
    iou_contact_evidence: float = 0.0         # IoU接触证据
    tracking_jitter_evidence: float = 0.0     # 跟踪抖动证据


def compute_risk_timeline(
    video_path: str,
    config: FileEventConfig,
    yolo_model: Any = None,
    tracker: Any = None,
) -> RiskTimelineResult:
    """
    计算单个mp4文件的风险时间序列

    Args:
        video_path: mp4文件路径
        config: 配置
        yolo_model: YOLO模型（可选，如果None则使用简化特征）
        tracker: 跟踪器（可选）

    Returns:
        RiskTimelineResult 包含风险时间序列和峰值
    """
    result = RiskTimelineResult()

    if not os.path.exists(video_path):
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # 采样间隔
    sample_interval = 1.0 / config.risk_sampling_fps
    sample_times = np.arange(0, duration, sample_interval)

    timestamps = []
    risk_scores = []

    prev_detections = None
    prev_time = None

    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # 计算当前帧的风险分数
        risk, detections = _compute_frame_risk(
            frame, t, prev_detections, prev_time, yolo_model, config
        )

        timestamps.append(t)
        risk_scores.append(risk)

        prev_detections = detections
        prev_time = t

    cap.release()

    result.timestamps = timestamps
    result.risk_scores = risk_scores

    # 找峰值
    if risk_scores:
        result.max_risk = max(risk_scores)
        result.max_risk_time = timestamps[risk_scores.index(result.max_risk)]

        # 找TopN峰值
        peaks = _find_peaks(
            timestamps, risk_scores,
            config.top_n_peaks,
            config.peak_min_distance_sec,
            config.peak_threshold
        )
        result.peak_times = [p[0] for p in peaks]
        result.peak_scores = [p[1] for p in peaks]

    return result


def _compute_frame_risk(
    frame: np.ndarray,
    timestamp: float,
    prev_detections: Optional[List[Dict]],
    prev_time: Optional[float],
    yolo_model: Any,
    config: FileEventConfig,
) -> Tuple[float, List[Dict]]:
    """
    计算单帧风险分数

    使用简化特征：
    1. 前景变化率（光流/帧差）
    2. 检测框密度和接近度
    3. 运动突变
    """
    h, w = frame.shape[:2]
    detections = []
    risk = 0.0

    # 如果有YOLO模型，使用检测结果
    if yolo_model is not None:
        try:
            results = yolo_model(frame, verbose=False)
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.5
                            cls = int(box.cls[0]) if hasattr(box, 'cls') else 0

                            cx = (xyxy[0] + xyxy[2]) / 2
                            cy = (xyxy[1] + xyxy[3]) / 2
                            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

                            detections.append({
                                "bbox": list(xyxy),
                                "center": (cx, cy),
                                "area": area,
                                "conf": conf,
                                "class": cls,
                            })
        except Exception:
            pass

    # 基于检测框计算风险
    if len(detections) >= 2:
        # 1. 最小距离风险
        min_dist = float('inf')
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                c1 = detections[i]["center"]
                c2 = detections[j]["center"]
                dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                min_dist = min(min_dist, dist)

        # 距离越近风险越高
        if min_dist < config.min_distance_threshold:
            risk += 0.4 * (1 - min_dist / config.min_distance_threshold)

        # 2. IoU接触风险
        max_iou = 0.0
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                iou = _compute_iou(detections[i]["bbox"], detections[j]["bbox"])
                max_iou = max(max_iou, iou)

        if max_iou > config.iou_contact_threshold:
            risk += 0.3 * min(1.0, max_iou / 0.2)

    # 3. 运动突变风险（基于前后帧对比）
    if prev_detections and prev_time is not None:
        dt = timestamp - prev_time
        if dt > 0 and len(detections) > 0 and len(prev_detections) > 0:
            # 匹配检测框并计算速度变化
            velocity_changes = []
            for curr in detections:
                best_match = None
                best_dist = float('inf')
                for prev in prev_detections:
                    dist = np.sqrt(
                        (curr["center"][0] - prev["center"][0])**2 +
                        (curr["center"][1] - prev["center"][1])**2
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_match = prev

                if best_match and best_dist < 200:  # 匹配阈值
                    velocity = best_dist / dt
                    velocity_changes.append(velocity)

            if velocity_changes:
                max_velocity = max(velocity_changes)
                # 高速度可能表示急刹或碰撞
                if max_velocity > 100:  # 像素/秒
                    risk += 0.3 * min(1.0, max_velocity / 500)

    return min(1.0, risk), detections


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个bbox的IoU"""
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


def _find_peaks(
    timestamps: List[float],
    values: List[float],
    top_n: int,
    min_distance: float,
    threshold: float,
) -> List[Tuple[float, float]]:
    """
    找到时间序列中的TopN峰值

    Returns:
        [(timestamp, value), ...] 按value降序排列
    """
    if not values:
        return []

    # 找所有局部最大值
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            if values[i] >= threshold:
                peaks.append((timestamps[i], values[i]))

    # 如果没有局部最大值，取全局最大值
    if not peaks:
        max_idx = np.argmax(values)
        if values[max_idx] >= threshold:
            peaks.append((timestamps[max_idx], values[max_idx]))

    # 按value降序排序
    peaks.sort(key=lambda x: x[1], reverse=True)

    # 非极大值抑制：去除距离太近的峰值
    filtered = []
    for peak in peaks:
        too_close = False
        for selected in filtered:
            if abs(peak[0] - selected[0]) < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(peak)
        if len(filtered) >= top_n:
            break

    return filtered


def compute_t0_validity(
    video_path: str,
    t0: float,
    config: FileEventConfig,
    yolo_model: Any = None,
    window_sec: float = 2.0,
) -> T0ValidityResult:
    """
    验证t0时刻是否像真正的碰撞发生点

    检查证据：
    1. t0附近两目标最小距离是否低于阈值/突降
    2. t0附近速度变化/jerk是否出现峰值
    3. t0附近框IoU/接触概率峰值
    4. t0附近跟踪抖动/ID switch

    Args:
        video_path: 视频路径
        t0: 待验证的碰撞时刻
        config: 配置
        yolo_model: YOLO模型
        window_sec: 检查窗口大小(秒)

    Returns:
        T0ValidityResult
    """
    result = T0ValidityResult(t0=t0)

    if not os.path.exists(video_path):
        result.t0_fallback = True
        result.t0_method = "fallback_file_not_found"
        result.validity_reason = "file_not_found"
        return result

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result.t0_fallback = True
        result.t0_method = "fallback_video_open_failed"
        result.validity_reason = "video_open_failed"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    # 采样t0附近的帧
    sample_times = []
    sample_step = 0.2  # 每0.2秒采样一次
    for dt in np.arange(-window_sec, window_sec + sample_step, sample_step):
        t = t0 + dt
        if 0 <= t <= duration:
            sample_times.append(t)

    if not sample_times:
        cap.release()
        result.t0_fallback = True
        result.t0_method = "fallback_no_valid_samples"
        result.validity_reason = "no_valid_samples"
        return result

    # 采集每帧的检测结果
    frame_data = []
    for t in sample_times:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        detections = []
        if yolo_model is not None:
            try:
                results = yolo_model(frame, verbose=False)
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                                xyxy = box.xyxy[0].cpu().numpy()
                                cx = (xyxy[0] + xyxy[2]) / 2
                                cy = (xyxy[1] + xyxy[3]) / 2
                                detections.append({
                                    "bbox": list(xyxy),
                                    "center": (cx, cy),
                                })
            except Exception:
                pass

        frame_data.append({
            "timestamp": t,
            "detections": detections,
        })

    cap.release()

    if not frame_data:
        result.t0_fallback = True
        result.t0_method = "fallback_no_frame_data"
        result.validity_reason = "no_frame_data"
        return result

    # 计算各项证据
    reasons = []

    # 1. 最小距离证据
    min_distances = []
    for fd in frame_data:
        dets = fd["detections"]
        if len(dets) >= 2:
            for i in range(len(dets)):
                for j in range(i + 1, len(dets)):
                    c1, c2 = dets[i]["center"], dets[j]["center"]
                    dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                    min_distances.append((fd["timestamp"], dist))

    if min_distances:
        # 找t0附近的最小距离
        t0_distances = [d for t, d in min_distances if abs(t - t0) < 0.5]
        if t0_distances:
            min_dist_at_t0 = min(t0_distances)
            if min_dist_at_t0 < config.min_distance_threshold:
                evidence = 1 - min_dist_at_t0 / config.min_distance_threshold
                result.min_distance_evidence = evidence
                if evidence > 0.3:
                    reasons.append(f"min_dist={min_dist_at_t0:.0f}px")

    # 2. 距离突降证据
    if len(min_distances) >= 3:
        sorted_by_time = sorted(min_distances, key=lambda x: x[0])
        for i in range(1, len(sorted_by_time)):
            t_curr, d_curr = sorted_by_time[i]
            t_prev, d_prev = sorted_by_time[i-1]

            if abs(t_curr - t0) < 0.5 and d_prev > 0:
                drop_ratio = (d_prev - d_curr) / d_prev
                if drop_ratio > config.distance_drop_threshold:
                    result.distance_drop_evidence = min(1.0, drop_ratio)
                    reasons.append(f"dist_drop={drop_ratio:.2f}")
                    break

    # 3. IoU接触证据
    max_iou_at_t0 = 0.0
    for fd in frame_data:
        if abs(fd["timestamp"] - t0) > 0.5:
            continue
        dets = fd["detections"]
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                iou = _compute_iou(dets[i]["bbox"], dets[j]["bbox"])
                max_iou_at_t0 = max(max_iou_at_t0, iou)

    if max_iou_at_t0 > config.iou_contact_threshold:
        result.iou_contact_evidence = min(1.0, max_iou_at_t0 / 0.2)
        reasons.append(f"iou={max_iou_at_t0:.2f}")

    # 计算综合validity
    validity = max(
        result.min_distance_evidence * 0.4,
        result.distance_drop_evidence * 0.3,
        result.iou_contact_evidence * 0.3,
        result.velocity_change_evidence * 0.2,
        result.tracking_jitter_evidence * 0.1,
    )

    result.validity = min(1.0, validity)
    result.t0_fallback = len(reasons) == 0
    result.t0_method = "evidence_based" if reasons else "fallback_no_evidence"
    result.validity_reason = "; ".join(reasons) if reasons else "no_collision_evidence"

    return result


def generate_file_candidates(
    video_path: str,
    config: FileEventConfig,
    yolo_model: Any = None,
    camera_id: str = "camera-1",
) -> List[Dict]:
    """
    为单个mp4文件生成候选clip列表

    Args:
        video_path: mp4文件路径
        config: 配置
        yolo_model: YOLO模型
        camera_id: 摄像头ID

    Returns:
        候选clip列表，每个包含：
        - clip_id, video_path, start_time, end_time
        - t_event, t0, t0_validity, validity_reason
        - coverage_effective, risk_peak
        - is_file_level, is_subclip
    """
    candidates = []

    if not os.path.exists(video_path):
        return candidates

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return candidates

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    file_basename = os.path.basename(video_path)
    file_id = os.path.splitext(file_basename)[0]

    # 1. 计算风险时间序列
    risk_result = compute_risk_timeline(video_path, config, yolo_model)

    # 2. 生成 file-level 候选（整个文件）
    file_candidate = {
        "clip_id": file_id,
        "video_path": video_path,
        "start_time": 0.0,
        "end_time": duration,
        "duration": duration,
        "camera_id": camera_id,
        "is_file_level": True,
        "is_subclip": False,
        "clip_source": "file_level",

        # 风险信息
        "risk_peak": risk_result.max_risk,
        "t_event": risk_result.max_risk_time,
        "peak_times": risk_result.peak_times,
        "peak_scores": risk_result.peak_scores,
    }

    # 3. 估计t0并验证
    if risk_result.max_risk > config.peak_threshold:
        t0_estimate = risk_result.max_risk_time
    else:
        t0_estimate = duration / 2.0  # fallback: 中点

    t0_result = compute_t0_validity(video_path, t0_estimate, config, yolo_model)

    file_candidate.update({
        "t0": t0_result.t0,
        "t0_fallback": t0_result.t0_fallback,
        "t0_method": t0_result.t0_method,
        "t0_validity": t0_result.validity,
        "validity_reason": t0_result.validity_reason,
    })

    # 4. 计算 coverage_effective
    pre_ok = min(1.0, t0_result.t0 / config.pre_roll) if config.pre_roll > 0 else 1.0
    post_ok = min(1.0, (duration - t0_result.t0) / config.post_roll) if config.post_roll > 0 else 1.0
    coverage_raw = pre_ok * post_ok
    coverage_effective = t0_result.validity * coverage_raw

    file_candidate.update({
        "pre_ok": pre_ok,
        "post_ok": post_ok,
        "coverage_raw": coverage_raw,
        "coverage_effective": coverage_effective,
        "late_start_penalty": max(0, config.pre_roll - t0_result.t0),
    })

    # 5. 计算基础分数和最终分数
    base_score = risk_result.max_risk
    # final_score = base + coverage_boost - penalty
    final_score = base_score + 0.3 * coverage_effective - 0.02 * file_candidate["late_start_penalty"]
    file_candidate["base_score"] = base_score
    file_candidate["clip_score"] = base_score  # 兼容旧字段
    file_candidate["final_score"] = final_score

    # 标记是否完整覆盖事故过程
    file_candidate["is_full_process"] = (
        pre_ok >= 0.8 and
        post_ok >= 0.8 and
        t0_result.validity >= 0.3
    )

    candidates.append(file_candidate)

    # 6. 如果模式允许，生成 event-window 子候选
    if config.candidate_mode == "file_plus_subclips" and risk_result.peak_times:
        subclips = _generate_subclips(
            video_path, duration, risk_result, config, yolo_model, camera_id
        )
        candidates.extend(subclips)

    return candidates


def _generate_subclips(
    video_path: str,
    duration: float,
    risk_result: RiskTimelineResult,
    config: FileEventConfig,
    yolo_model: Any,
    camera_id: str,
) -> List[Dict]:
    """
    围绕风险峰值生成子候选窗口
    """
    subclips = []
    file_id = os.path.splitext(os.path.basename(video_path))[0]

    # 生成初始窗口
    windows = []
    for i, (t_peak, score) in enumerate(zip(risk_result.peak_times, risk_result.peak_scores)):
        start = max(0, t_peak - config.pre_roll)
        end = min(duration, t_peak + config.post_roll)
        windows.append({
            "start": start,
            "end": end,
            "t_event": t_peak,
            "score": score,
        })

    # 合并重叠窗口
    windows.sort(key=lambda x: x["start"])
    merged = []
    for w in windows:
        if merged and w["start"] <= merged[-1]["end"] + config.merge_gap_sec:
            # 合并
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["score"] = max(merged[-1]["score"], w["score"])
            # t_event取分数更高的
            if w["score"] > merged[-1]["score"]:
                merged[-1]["t_event"] = w["t_event"]
        else:
            merged.append(w.copy())

    # 为每个合并窗口创建子候选
    for i, w in enumerate(merged):
        subclip_id = f"{file_id}_sub{i}"
        t0_estimate = w["t_event"]

        # 验证t0
        t0_result = compute_t0_validity(video_path, t0_estimate, config, yolo_model)

        # 计算coverage_effective（相对于子窗口）
        sub_duration = w["end"] - w["start"]
        t0_in_subclip = t0_result.t0 - w["start"]

        pre_ok = min(1.0, t0_in_subclip / config.pre_roll) if config.pre_roll > 0 else 1.0
        post_ok = min(1.0, (sub_duration - t0_in_subclip) / config.post_roll) if config.post_roll > 0 else 1.0
        coverage_raw = pre_ok * post_ok
        coverage_effective = t0_result.validity * coverage_raw

        base_score = w["score"]
        final_score = base_score + 0.3 * coverage_effective

        subclip = {
            "clip_id": subclip_id,
            "video_path": video_path,
            "start_time": w["start"],
            "end_time": w["end"],
            "duration": sub_duration,
            "camera_id": camera_id,
            "is_file_level": False,
            "is_subclip": True,
            "clip_source": "event_window",

            "risk_peak": w["score"],
            "t_event": w["t_event"],
            "t0": t0_result.t0,
            "t0_fallback": t0_result.t0_fallback,
            "t0_method": t0_result.t0_method,
            "t0_validity": t0_result.validity,
            "validity_reason": t0_result.validity_reason,

            "pre_ok": pre_ok,
            "post_ok": post_ok,
            "coverage_raw": coverage_raw,
            "coverage_effective": coverage_effective,
            "late_start_penalty": 0.0,  # 子窗口无此惩罚

            "base_score": base_score,
            "clip_score": base_score,
            "final_score": final_score,

            "is_full_process": (
                pre_ok >= 0.8 and
                post_ok >= 0.8 and
                t0_result.validity >= 0.3
            ),
        }
        subclips.append(subclip)

    return subclips


def apply_conditional_retention(
    candidate: Dict,
    verdict: str,
    config: FileEventConfig,
) -> Tuple[bool, str]:
    """
    应用有条件保留策略

    规则：
    - accident_score=1.0 且 verdict ∈ {YES, POST_EVENT_ONLY, UNCERTAIN} → kept=True
    - accident_score=1.0 且 verdict==NO：
        仅当满足任一条件才 kept=True：
        a) t0_validity >= validity_threshold
        b) risk_peak >= peak_threshold
        c) roi_median_edge >= roi_threshold (如果有ROI信息)

    Args:
        candidate: 候选clip信息
        verdict: VLM verdict
        config: 配置

    Returns:
        (kept, keep_reason)
    """
    accident_score = candidate.get("accident_score", 0.0)
    t0_validity = candidate.get("t0_validity", 0.0)
    risk_peak = candidate.get("risk_peak", 0.0)
    roi_median = candidate.get("roi_median_edge", 0.0)

    verdict_upper = verdict.upper() if verdict else "NO"

    # 规则1: 非NO的verdict直接保留
    if verdict_upper in ("YES", "POST_EVENT_ONLY", "UNCERTAIN"):
        if verdict_upper == "YES":
            return True, f"verdict={verdict_upper}_confirmed"
        elif verdict_upper == "POST_EVENT_ONLY":
            return True, f"verdict={verdict_upper}_post_event"
        else:
            return True, f"verdict={verdict_upper}_needs_review"

    # 规则2: verdict==NO 时的有条件保留
    if verdict_upper == "NO":
        reasons = []

        # 检查各项条件
        if t0_validity >= config.validity_threshold:
            reasons.append(f"validity={t0_validity:.2f}>={config.validity_threshold}")

        if risk_peak >= config.risk_peak_threshold:
            reasons.append(f"risk_peak={risk_peak:.2f}>={config.risk_peak_threshold}")

        if roi_median >= config.roi_median_threshold:
            reasons.append(f"roi_median={roi_median:.0f}>={config.roi_median_threshold}")

        if reasons:
            return True, f"NO_but_kept: {'; '.join(reasons)}"
        else:
            return False, f"NO_dropped: validity={t0_validity:.2f}, peak={risk_peak:.2f}"

    return False, f"unknown_verdict={verdict}"


def compute_rank_score(
    candidate: Dict,
    pre_roll: float = 8.0,
    post_roll: float = 12.0,
    B: float = 0.20,
    P: float = 0.20,
    confirm_bonus: float = 0.10,
    uncertain_bonus: float = 0.05,
) -> Dict:
    """
    计算单一排序主分 rank_score（语义对齐版）

    公式:
        rank_score = final_score + full_process_bonus - post_event_penalty + confirm_bonus

    其中:
        full_process_score = pre_score * impact_score * post_score (仅当verdict=YES时)
        post_event_score = 1 if verdict=POST_EVENT_ONLY else 0
        full_process_bonus = B * full_process_score
        post_event_penalty = P * post_event_score
        confirm_bonus = verdict加分

    语义约束:
        - verdict=POST_EVENT_ONLY => full_process_score=0, post_event_score=1
        - verdict=YES => full_process_score根据覆盖度计算, post_event_score=0
        - verdict=NO/UNCERTAIN => full_process_score=0, post_event_score=0

    Args:
        candidate: 候选dict
        pre_roll: 碰撞前目标覆盖时长
        post_roll: 碰撞后目标覆盖时长
        B: full_process_bonus权重
        P: post_event_penalty权重
        confirm_bonus: verdict=YES加分
        uncertain_bonus: verdict=UNCERTAIN加分

    Returns:
        更新后的candidate dict（包含所有评分字段）
    """
    duration = candidate.get("duration", 1.0)
    t0 = candidate.get("t0", 0.0)
    t0_validity = candidate.get("t0_validity", 0.0)
    final_score = candidate.get("final_score", 0.0)
    verdict = candidate.get("verdict", "").upper()

    # ===== 1. 计算 pre_score / post_score =====
    pre_score = min(1.0, t0 / pre_roll) if pre_roll > 0 else 0.0
    post_score = min(1.0, (duration - t0) / post_roll) if post_roll > 0 else 0.0

    # ===== 2. 计算 impact_score = t0_validity =====
    impact_score = t0_validity

    # ===== 3. 语义对齐：根据verdict决定评分 =====
    if verdict == "YES":
        # 完整覆盖事故过程
        full_process_score = pre_score * impact_score * post_score
        post_event_score = 0.0
        verdict_bonus = confirm_bonus
    elif verdict == "POST_EVENT_ONLY":
        # 仅事故后果，无完整过程
        full_process_score = 0.0
        post_event_score = 1.0
        verdict_bonus = uncertain_bonus  # 仍有一定价值
    elif verdict == "UNCERTAIN":
        # 不确定，需人工复核
        full_process_score = 0.0
        post_event_score = 0.0
        verdict_bonus = uncertain_bonus
    else:
        # NO 或未知
        full_process_score = 0.0
        post_event_score = 0.0
        verdict_bonus = 0.0

    # ===== 4. 计算 bonus/penalty =====
    full_process_bonus = B * full_process_score
    post_event_penalty = P * post_event_score

    # ===== 5. 计算 rank_score =====
    rank_score = final_score + full_process_bonus - post_event_penalty + verdict_bonus

    # ===== 6. 更新candidate字段 =====
    candidate["pre_score"] = pre_score
    candidate["impact_score"] = impact_score
    candidate["post_score"] = post_score
    candidate["full_process_score"] = full_process_score
    candidate["post_event_score"] = post_event_score
    candidate["full_process_bonus"] = full_process_bonus
    candidate["post_event_penalty"] = post_event_penalty
    candidate["verdict_bonus"] = verdict_bonus
    candidate["rank_score"] = rank_score

    return candidate


def rank_candidates(
    candidates: List[Dict],
    pre_roll: float = 8.0,
    post_roll: float = 12.0,
    B: float = 0.20,
    P: float = 0.20,
    confirm_bonus: float = 0.10,
    uncertain_bonus: float = 0.05,
) -> List[Dict]:
    """
    对候选列表排序（单一rank_score版）

    排序规则:
        按 rank_score 降序排列（唯一排序字段，禁止tuple sort_key）

    Args:
        candidates: 候选列表
        pre_roll: 碰撞前目标覆盖时长
        post_roll: 碰撞后目标覆盖时长
        B: full_process_bonus权重
        P: post_event_penalty权重
        confirm_bonus: verdict=YES加分
        uncertain_bonus: verdict=UNCERTAIN加分

    Returns:
        排序后的候选列表（已更新rank_score字段）
    """
    # 为每个候选计算rank_score
    for c in candidates:
        compute_rank_score(
            c,
            pre_roll=pre_roll,
            post_roll=post_roll,
            B=B,
            P=P,
            confirm_bonus=confirm_bonus,
            uncertain_bonus=uncertain_bonus,
        )

    # 单一字段排序（禁止tuple sort_key）
    return sorted(candidates, key=lambda c: c.get("rank_score", 0.0), reverse=True)


def log_candidates_ranking(candidates: List[Dict], logger=None):
    """输出候选排名日志（rank_score版）"""
    log_fn = logger.info if logger else print

    log_fn("\n[FileEventLocator] 候选排名（按 rank_score 降序）:")
    log_fn("-" * 160)
    log_fn(
        f"{'Rank':>4} | {'Clip ID':<16} | {'verdict':<16} | {'t0':>5} | "
        f"{'pre':>5} | {'imp':>5} | {'post':>5} | {'full_p':>6} | {'post_e':>6} | "
        f"{'final':>7} | {'bonus':>6} | {'penalty':>7} | {'rank_score':>10}"
    )
    log_fn("-" * 160)

    for i, c in enumerate(candidates, 1):
        clip_id = c.get("clip_id", "N/A")[:16]
        verdict = c.get("verdict", "-")[:16]
        t0 = c.get("t0", 0)
        pre = c.get("pre_score", 0)
        imp = c.get("impact_score", 0)
        post = c.get("post_score", 0)
        full_p = c.get("full_process_score", 0)
        post_e = c.get("post_event_score", 0)
        final = c.get("final_score", 0)
        bonus = c.get("full_process_bonus", 0)
        penalty = c.get("post_event_penalty", 0)
        rank = c.get("rank_score", 0)

        log_fn(
            f"{i:>4} | {clip_id:<16} | {verdict:<16} | {t0:>5.1f} | "
            f"{pre:>5.2f} | {imp:>5.2f} | {post:>5.2f} | {full_p:>6.3f} | {post_e:>6.1f} | "
            f"{final:>7.4f} | {bonus:>6.3f} | {penalty:>7.3f} | {rank:>10.4f}"
        )

    log_fn("-" * 160)
