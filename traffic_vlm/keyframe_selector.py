"""
关键帧选择模块 - 基于信号的智能抽帧策略

目标：为VLM选择最具信息量的帧，覆盖事故的"前-中-后"因果链。

信号类型：
- motion_peak: 运动峰值（bbox位移和变化率）
- interaction_peak: 交互峰值（目标间最小距离变化）
- trajectory_break: 轨迹中断（track丢失/ID切换/突然停止）
- post_event_cue: 事故后线索（异常停车、聚集等）

抽帧模式：
- FAST: 4-6帧，用于快速验证
- ESCALATED: 12-16帧，用于升级验证
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from .config import ProgressiveVLMConfig


@dataclass
class FrameSignal:
    """单帧信号数据"""
    frame_idx: int
    timestamp: float
    motion_score: float = 0.0           # 运动峰值分数
    interaction_score: float = 0.0      # 交互峰值分数
    trajectory_break_score: float = 0.0 # 轨迹中断分数
    post_event_score: float = 0.0       # 事故后线索分数
    combined_score: float = 0.0         # 综合分数
    reason_tags: List[str] = field(default_factory=list)

    # 原始检测数据（用于元数据包构建）
    detections: List[Dict] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    min_pair_distance: Optional[float] = None
    nearest_pair_ids: Optional[Tuple[int, int]] = None


@dataclass
class FrameRequest:
    """帧选择请求结果"""
    frame_idx: int
    timestamp: float
    reason_tags: List[str]
    score_components: Dict[str, float]
    priority: int = 0  # 0=context, 1=peak, 2=post

    # 关联的检测数据
    detections: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeyframeSelector:
    """关键帧选择器

    基于YOLO检测和跟踪结果，选择最具信息量的帧。
    不依赖VLM，使用廉价信号进行选择。
    """

    def __init__(self, config: ProgressiveVLMConfig):
        self.config = config
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        self.person_classes = {0}  # person
        self.bike_classes = {1, 3}  # bicycle, motorcycle

    def compute_frame_signals(
        self,
        frame_results: List[Dict],
        tracks: Dict[int, Dict],
        fps: float = 30.0,
        frame_timestamps: Optional[List[float]] = None,
        clip_duration: float = 0.0,
    ) -> List[FrameSignal]:
        """计算每帧的信号分数

        Args:
            frame_results: YOLO检测结果列表，每个元素对应一帧
            tracks: 跟踪结果，{track_id: {category, trajectory}}
            fps: 视频帧率
            frame_timestamps: 每帧的实际时间戳（相对于clip起点，秒）
            clip_duration: clip时长（秒），用于计算均匀分布时间戳

        Returns:
            每帧的信号数据列表
        """
        if not frame_results:
            return []

        n_frames = len(frame_results)
        signals = []

        # [A1修复] 如果没有提供实际时间戳，根据clip_duration计算均匀分布的时间戳
        if frame_timestamps is None or len(frame_timestamps) != n_frames:
            if clip_duration > 0 and n_frames > 1:
                # 使用均匀分布的时间戳（相对于clip起点，覆盖0到clip_duration）
                frame_timestamps = [i * clip_duration / (n_frames - 1) for i in range(n_frames)]
            else:
                # 回退到旧逻辑
                frame_timestamps = [fidx / fps for fidx in range(n_frames)]

        # 预计算每帧的检测框中心
        frame_centers = []  # [(frame_idx, [(track_id, cx, cy, cls)])]

        for fidx, fr in enumerate(frame_results):
            centers = []
            detections = fr.get("detections", [])
            for det in detections:
                bbox = det.get("bbox", [0, 0, 0, 0])
                track_id = det.get("track_id", -1)
                cls = det.get("class", -1)
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                centers.append((track_id, cx, cy, cls, bbox))
            frame_centers.append((fidx, centers))

        # 计算每帧信号
        for fidx in range(n_frames):
            timestamp = frame_timestamps[fidx]  # [A1修复] 使用实际时间戳
            fr = frame_results[fidx]
            detections = fr.get("detections", [])

            signal = FrameSignal(
                frame_idx=fidx,
                timestamp=timestamp,
                detections=detections,
                track_ids=[d.get("track_id", -1) for d in detections]
            )

            # 1. 运动峰值：计算bbox位移
            motion_score = self._compute_motion_score(fidx, frame_centers)
            signal.motion_score = motion_score
            if motion_score > 0.5:
                signal.reason_tags.append("motion_peak")

            # 2. 交互峰值：计算目标间最小距离
            interaction_score, min_dist, pair_ids = self._compute_interaction_score(
                fidx, frame_centers
            )
            signal.interaction_score = interaction_score
            signal.min_pair_distance = min_dist
            signal.nearest_pair_ids = pair_ids
            if interaction_score > 0.5:
                signal.reason_tags.append("interaction_peak")

            # 3. 轨迹中断：检测track丢失/出现
            break_score = self._compute_trajectory_break_score(fidx, frame_centers, tracks)
            signal.trajectory_break_score = break_score
            if break_score > 0.3:
                signal.reason_tags.append("trajectory_break")

            # 4. 事故后线索：检测异常停车
            post_score = self._compute_post_event_score(fidx, frame_centers, tracks, fps)
            signal.post_event_score = post_score
            if post_score > 0.3:
                signal.reason_tags.append("post_event_cue")

            # 综合分数（加权）
            signal.combined_score = (
                self.config.motion_peak_weight * motion_score +
                self.config.interaction_peak_weight * interaction_score +
                self.config.trajectory_break_weight * break_score +
                self.config.post_event_cue_weight * post_score
            )

            signals.append(signal)

        return signals

    def _compute_motion_score(
        self,
        fidx: int,
        frame_centers: List[Tuple[int, List]],
    ) -> float:
        """计算运动峰值分数（基于bbox位移）"""
        if fidx == 0 or fidx >= len(frame_centers):
            return 0.0

        _, curr_centers = frame_centers[fidx]
        _, prev_centers = frame_centers[fidx - 1]

        if not curr_centers or not prev_centers:
            return 0.0

        # 匹配相同track_id的目标，计算位移
        prev_map = {c[0]: (c[1], c[2]) for c in prev_centers if c[0] >= 0}

        displacements = []
        for track_id, cx, cy, cls, bbox in curr_centers:
            if track_id >= 0 and track_id in prev_map:
                px, py = prev_map[track_id]
                disp = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                displacements.append(disp)

        if not displacements:
            return 0.0

        # 归一化到0-1（假设最大位移约100像素/帧）
        max_disp = max(displacements)
        avg_disp = np.mean(displacements)

        # 使用sigmoid软归一化
        score = 1.0 / (1.0 + np.exp(-0.05 * (max_disp - 30)))
        return float(score)

    def _compute_interaction_score(
        self,
        fidx: int,
        frame_centers: List[Tuple[int, List]],
    ) -> Tuple[float, Optional[float], Optional[Tuple[int, int]]]:
        """计算交互峰值分数（基于目标间最小距离）

        Returns:
            (score, min_distance, (id1, id2))
        """
        if fidx >= len(frame_centers):
            return 0.0, None, None

        _, centers = frame_centers[fidx]

        # 筛选车辆和行人/非机动车
        vehicles = [(c[0], c[1], c[2], c[4]) for c in centers
                   if c[3] in self.vehicle_classes and c[0] >= 0]
        vrus = [(c[0], c[1], c[2], c[4]) for c in centers
               if (c[3] in self.person_classes or c[3] in self.bike_classes) and c[0] >= 0]

        min_dist = float('inf')
        nearest_pair = None

        # 计算车-车距离
        for i, (id1, cx1, cy1, bbox1) in enumerate(vehicles):
            for j, (id2, cx2, cy2, bbox2) in enumerate(vehicles):
                if i >= j:
                    continue
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                # 考虑bbox大小，计算边界距离
                w1 = bbox1[2] - bbox1[0]
                w2 = bbox2[2] - bbox2[0]
                edge_dist = max(0, dist - (w1 + w2) / 2)
                if edge_dist < min_dist:
                    min_dist = edge_dist
                    nearest_pair = (id1, id2)

        # 计算车-VRU距离
        for id1, cx1, cy1, bbox1 in vehicles:
            for id2, cx2, cy2, bbox2 in vrus:
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                w1 = bbox1[2] - bbox1[0]
                w2 = bbox2[2] - bbox2[0]
                edge_dist = max(0, dist - (w1 + w2) / 2)
                if edge_dist < min_dist:
                    min_dist = edge_dist
                    nearest_pair = (id1, id2)

        if min_dist == float('inf'):
            return 0.0, None, None

        # 归一化：距离越小，分数越高
        # 距离<50像素时分数>0.5
        score = 1.0 / (1.0 + min_dist / 50.0)
        return float(score), float(min_dist), nearest_pair

    def _compute_trajectory_break_score(
        self,
        fidx: int,
        frame_centers: List[Tuple[int, List]],
        tracks: Dict[int, Dict],
    ) -> float:
        """计算轨迹中断分数（track出现/消失）"""
        if fidx == 0 or fidx >= len(frame_centers):
            return 0.0

        _, curr_centers = frame_centers[fidx]
        _, prev_centers = frame_centers[fidx - 1]

        curr_ids = {c[0] for c in curr_centers if c[0] >= 0}
        prev_ids = {c[0] for c in prev_centers if c[0] >= 0}

        # 消失的track
        disappeared = prev_ids - curr_ids
        # 新出现的track
        appeared = curr_ids - prev_ids

        # 中断数量
        break_count = len(disappeared) + len(appeared)

        if break_count == 0:
            return 0.0

        # 归一化（假设最多5个中断）
        score = min(1.0, break_count / 3.0)
        return float(score)

    def _compute_post_event_score(
        self,
        fidx: int,
        frame_centers: List[Tuple[int, List]],
        tracks: Dict[int, Dict],
        fps: float,
    ) -> float:
        """计算事故后线索分数（异常停车/低速）"""
        if fidx < 5 or fidx >= len(frame_centers):
            return 0.0

        _, curr_centers = frame_centers[fidx]

        # 检查近5帧的运动情况
        vehicle_ids = [c[0] for c in curr_centers
                      if c[3] in self.vehicle_classes and c[0] >= 0]

        if not vehicle_ids:
            return 0.0

        slow_count = 0
        for vid in vehicle_ids:
            # 检查该track在最近5帧的位移
            positions = []
            for i in range(max(0, fidx - 4), fidx + 1):
                _, centers = frame_centers[i]
                for c in centers:
                    if c[0] == vid:
                        positions.append((c[1], c[2]))
                        break

            if len(positions) >= 3:
                # 计算总位移
                total_disp = 0
                for j in range(1, len(positions)):
                    dx = positions[j][0] - positions[j-1][0]
                    dy = positions[j][1] - positions[j-1][1]
                    total_disp += np.sqrt(dx**2 + dy**2)

                # 平均位移<5像素/帧 视为停止/低速
                avg_disp = total_disp / (len(positions) - 1)
                if avg_disp < 5:
                    slow_count += 1

        if slow_count == 0:
            return 0.0

        # 归一化
        score = min(1.0, slow_count / 2.0)
        return float(score)

    def select_frames_for_clip(
        self,
        frame_results: List[Dict],
        tracks: Dict[int, Dict],
        clip_start_time: float,
        clip_duration: float,
        fps: float = 30.0,
        mode: str = "FAST",
        frame_timestamps: Optional[List[float]] = None,
        coverage_min: float = 0.5,
    ) -> List[FrameRequest]:
        """为clip选择关键帧（v2自适应帧预算 + 覆盖守卫）

        Args:
            frame_results: YOLO检测结果
            tracks: 跟踪结果
            clip_start_time: clip起始时间（秒）
            clip_duration: clip时长（秒）
            fps: 帧率
            mode: "FAST" 或 "ESCALATED"
            frame_timestamps: 帧的实际时间戳（相对clip起点）
            coverage_min: 最小覆盖率阈值（默认50%）

        Returns:
            选中的帧请求列表
        """
        # [A1修复] 计算信号，传入clip_duration以计算正确时间戳
        signals = self.compute_frame_signals(
            frame_results, tracks, fps,
            frame_timestamps=frame_timestamps,
            clip_duration=clip_duration
        )

        if not signals:
            return []

        n_frames = len(signals)

        # ===== v2自适应帧预算 =====
        if self.config.version == "v2":
            if mode == "FAST":
                target_frames = self.config.get_s1_frames(clip_duration)
            else:  # ESCALATED
                target_frames = self.config.get_s2_frames(clip_duration)
            # 限制不超过可用帧数的80%
            target_frames = min(target_frames, int(n_frames * 0.8))
            target_frames = max(4, target_frames)  # 至少4帧
        else:
            # v1兼容模式
            if mode == "FAST":
                min_frames = self.config.fast_frames_min
                max_frames = self.config.fast_frames_max
            else:
                min_frames = self.config.escalated_frames_min
                max_frames = self.config.escalated_frames_max
            target_frames = min(max_frames, max(min_frames, n_frames // 4))

        print(f"[KeyframeSelector] v={self.config.version} mode={mode} "
              f"duration={clip_duration:.1f}s n_frames={n_frames} target={target_frames}")

        # ===== [D修复] v2帧选择策略：uniform为主、event为辅 =====
        selected: List[FrameRequest] = []
        selected_indices: set = set()

        # 1. [D修复] 均匀分布基础帧（占60%配额）- 必须覆盖clip全程
        uniform_count = max(4, int(target_frames * 0.6))

        # 首尾帧必选
        first_idx = 0
        last_idx = n_frames - 1

        if first_idx not in selected_indices:
            selected.append(self._create_frame_request(
                signals[first_idx], priority=0, reason_override=["uniform_first"]
            ))
            selected_indices.add(first_idx)

        if last_idx not in selected_indices and last_idx != first_idx:
            selected.append(self._create_frame_request(
                signals[last_idx], priority=0, reason_override=["uniform_last"]
            ))
            selected_indices.add(last_idx)

        # 中间帧均匀分布
        remaining_uniform = uniform_count - len(selected)
        if remaining_uniform > 0 and n_frames > 2:
            # 使用linspace确保均匀覆盖
            uniform_positions = np.linspace(1, n_frames - 2, remaining_uniform)
            for pos in uniform_positions:
                idx = int(round(pos))
                idx = max(0, min(n_frames - 1, idx))
                if idx not in selected_indices:
                    selected.append(self._create_frame_request(
                        signals[idx], priority=0, reason_override=["uniform_base"]
                    ))
                    selected_indices.add(idx)

        # 2. 事件加帧：在高信号帧附近增加采样
        if self.config.version == "v2" and self.config.event_boost_enabled:
            # 找到高信号帧
            sorted_by_score = sorted(signals, key=lambda s: s.combined_score, reverse=True)
            event_frames = [s for s in sorted_by_score
                          if s.combined_score >= self.config.event_signal_threshold]

            # 取top事件帧
            boost_budget = min(self.config.event_boost_frames, target_frames - len(selected))
            neighbor_frames = int(self.config.neighbor_sec * fps)

            for event_sig in event_frames[:2]:  # 最多关注2个事件点
                if boost_budget <= 0:
                    break

                event_idx = event_sig.frame_idx

                # 在事件附近选择帧（前后各neighbor_frames帧）
                for offset in [-neighbor_frames, 0, neighbor_frames]:
                    if boost_budget <= 0:
                        break
                    idx = event_idx + offset
                    if 0 <= idx < n_frames and idx not in selected_indices:
                        reason = "event_before" if offset < 0 else (
                            "event_peak" if offset == 0 else "event_after"
                        )
                        selected.append(self._create_frame_request(
                            signals[idx], priority=1, reason_override=[reason]
                        ))
                        selected_indices.add(idx)
                        boost_budget -= 1

        # 3. 填充剩余配额：选择综合分数最高的帧
        remaining = target_frames - len(selected)
        if remaining > 0:
            sorted_by_score = sorted(signals, key=lambda s: s.combined_score, reverse=True)
            min_gap = max(1, n_frames // (target_frames * 2))

            for sig in sorted_by_score:
                if remaining <= 0:
                    break
                if sig.frame_idx in selected_indices:
                    continue
                # 检查间隔
                too_close = any(abs(sig.frame_idx - idx) < min_gap for idx in selected_indices)
                if too_close:
                    continue

                selected.append(self._create_frame_request(sig, priority=1))
                selected_indices.add(sig.frame_idx)
                remaining -= 1

        # 4. [D修复] Coverage守卫：检查时间跨度是否覆盖足够范围
        if selected and clip_duration > 0:
            timestamps = [r.timestamp for r in selected]
            ts_min, ts_max = min(timestamps), max(timestamps)
            span_pct = (ts_max - ts_min) / clip_duration

            if span_pct < coverage_min:
                print(f"[KeyframeSelector] ⚠ 覆盖不足 span={span_pct*100:.1f}% < {coverage_min*100:.0f}%，补充缺口帧")

                # 找出缺口时间段并补充帧
                # 如果开头缺口（ts_min > clip_duration * 0.1）
                if ts_min > clip_duration * 0.1:
                    early_idx = max(0, int(n_frames * 0.05))
                    if early_idx not in selected_indices:
                        selected.append(self._create_frame_request(
                            signals[early_idx], priority=0, reason_override=["coverage_gap_fill"]
                        ))
                        selected_indices.add(early_idx)

                # 如果结尾缺口（ts_max < clip_duration * 0.9）
                if ts_max < clip_duration * 0.9:
                    late_idx = min(n_frames - 1, int(n_frames * 0.95))
                    if late_idx not in selected_indices:
                        selected.append(self._create_frame_request(
                            signals[late_idx], priority=2, reason_override=["coverage_gap_fill"]
                        ))
                        selected_indices.add(late_idx)

        # 按时间排序
        selected.sort(key=lambda r: r.timestamp)

        # 计算最终覆盖率
        if selected and clip_duration > 0:
            timestamps = [r.timestamp for r in selected]
            ts_min, ts_max = min(timestamps), max(timestamps)
            final_span_pct = (ts_max - ts_min) / clip_duration * 100
        else:
            final_span_pct = 0

        print(f"[KeyframeSelector] 最终选帧: {len(selected)}帧, "
              f"indices={sorted([r.frame_idx for r in selected])}, "
              f"span={final_span_pct:.1f}%")

        return selected

    def _create_frame_request(
        self,
        signal: FrameSignal,
        priority: int,
        reason_override: Optional[List[str]] = None,
    ) -> FrameRequest:
        """从信号创建帧请求"""
        reasons = reason_override if reason_override else signal.reason_tags.copy()
        if not reasons:
            reasons = ["default"]

        return FrameRequest(
            frame_idx=signal.frame_idx,
            timestamp=signal.timestamp,
            reason_tags=reasons,
            score_components={
                "motion": signal.motion_score,
                "interaction": signal.interaction_score,
                "trajectory_break": signal.trajectory_break_score,
                "post_event": signal.post_event_score,
                "combined": signal.combined_score,
            },
            priority=priority,
            detections=signal.detections,
            metadata={
                "min_pair_distance": signal.min_pair_distance,
                "nearest_pair_ids": signal.nearest_pair_ids,
                "track_ids": signal.track_ids,
            }
        )


def build_metadata_pack(
    frame_requests: List[FrameRequest],
    frame_results: List[Dict],
    tracks: Dict[int, Dict],
    config: ProgressiveVLMConfig,
) -> str:
    """构建结构化文本元数据包

    为VLM提供结构化的检测/轨迹信息，替代图像叠加框。

    Args:
        frame_requests: 选中的帧请求
        frame_results: 完整的YOLO检测结果
        tracks: 跟踪结果
        config: 配置

    Returns:
        结构化文本元数据（JSON格式字符串）
    """
    if not config.include_object_metadata_text:
        return ""

    metadata_lines = ["【检测元数据】"]

    # 类别映射
    cls_names = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck"
    }

    for req in frame_requests:
        frame_idx = req.frame_idx
        if frame_idx >= len(frame_results):
            continue

        fr = frame_results[frame_idx]
        detections = fr.get("detections", [])

        # 构建帧级元数据
        frame_meta = f"\n[帧{frame_idx}] t={req.timestamp:.2f}s"
        if req.reason_tags:
            frame_meta += f" ({','.join(req.reason_tags)})"
        metadata_lines.append(frame_meta)

        # 检测目标列表
        obj_lines = []
        for det in detections:
            track_id = det.get("track_id", -1)
            cls = det.get("class", -1)
            conf = det.get("confidence", 0)
            bbox = det.get("bbox", [0, 0, 0, 0])

            cls_name = cls_names.get(cls, f"cls{cls}")
            # 归一化bbox（假设图像尺寸1920x1080）
            norm_bbox = [
                bbox[0] / 1920, bbox[1] / 1080,
                (bbox[2] - bbox[0]) / 1920, (bbox[3] - bbox[1]) / 1080
            ]
            obj_lines.append(
                f"  ID:{track_id} {cls_name} conf={conf:.2f} "
                f"bbox=[{norm_bbox[0]:.3f},{norm_bbox[1]:.3f},{norm_bbox[2]:.3f},{norm_bbox[3]:.3f}]"
            )

        if obj_lines:
            metadata_lines.extend(obj_lines[:10])  # 最多10个目标

        # 添加交互信息
        if req.metadata.get("min_pair_distance") is not None:
            min_dist = req.metadata["min_pair_distance"]
            pair_ids = req.metadata.get("nearest_pair_ids")
            if pair_ids:
                metadata_lines.append(
                    f"  ⚠ 最近距离: {min_dist:.1f}px (ID:{pair_ids[0]}-ID:{pair_ids[1]})"
                )

    # 添加轨迹摘要
    if tracks:
        metadata_lines.append("\n【轨迹摘要】")
        for tid, info in list(tracks.items())[:5]:  # 最多5条轨迹
            traj = info.get("trajectory", [])
            category = info.get("category", "unknown")
            if len(traj) >= 2:
                start = traj[0]
                end = traj[-1]
                metadata_lines.append(
                    f"  ID:{tid} {category}: ({start[1]:.0f},{start[2]:.0f})→({end[1]:.0f},{end[2]:.0f})"
                )

    return "\n".join(metadata_lines)
