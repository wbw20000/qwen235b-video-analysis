"""
轨迹碰撞评分模块

基于目标检测和轨迹跟踪计算三个评分指标：
1. collision_score - 碰撞风险（IOU/中心距离）
2. track_intersection_score - 轨迹交叉检测
3. deceleration_score - 急刹/速度突变检测

这些评分会被写入候选帧的metadata，供后续聚类时使用。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .config import TrajectoryScoreConfig


def compute_trajectory_scores(
    tracks: Dict[int, Dict],
    frame_results: List[Dict],
    config: "TrajectoryScoreConfig"
) -> Dict[str, float]:
    """
    基于轨迹计算三个评分

    Args:
        tracks: 跟踪结果 {track_id: {"category": int, "trajectory": [(ts, x1, y1, x2, y2), ...]}}
        frame_results: 每帧检测结果 [{"timestamp": float, "detections": [...]}]
        config: 评分配置

    Returns:
        {
            "collision_score": float,           # 0~1，碰撞风险
            "track_intersection_score": float,  # 0~1，轨迹交叉
            "deceleration_score": float         # 0~1，急刹检测
        }
    """
    scores = {
        "collision_score": 0.0,
        "track_intersection_score": 0.0,
        "deceleration_score": 0.0,
    }

    # 至少需要2个目标才能计算碰撞/交叉
    if not tracks or len(tracks) < 2:
        # 但仍可计算单目标的急刹
        if tracks:
            scores["deceleration_score"] = _compute_deceleration_score(tracks, config)
        return scores

    # 1. collision_score - 基于IOU/距离
    scores["collision_score"] = _compute_collision_score(frame_results, config)

    # 2. track_intersection_score - 轨迹交叉
    scores["track_intersection_score"] = _compute_intersection_score(tracks)

    # 3. deceleration_score - 急刹检测
    scores["deceleration_score"] = _compute_deceleration_score(tracks, config)

    return scores


def _compute_collision_score(frame_results: List[Dict], config: "TrajectoryScoreConfig") -> float:
    """
    计算碰撞分数 - 基于每帧的目标IOU/距离

    遍历所有帧，计算目标两两之间的IOU和中心距离，取最大分数。
    """
    max_score = 0.0

    for frame in frame_results:
        detections = frame.get("detections", [])
        if len(detections) < 2:
            continue

        # 两两计算IOU和距离
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                box_i = detections[i].get("bbox", [0, 0, 0, 0])
                box_j = detections[j].get("bbox", [0, 0, 0, 0])

                # 计算IOU
                iou = _compute_iou(box_i, box_j)
                if iou > config.collision_iou_threshold:
                    # IOU高说明已经重叠/碰撞
                    score = min(1.0, iou / 0.5)  # 归一化到0~1
                    max_score = max(max_score, score)
                else:
                    # 基于中心距离评估接近程度
                    dist = _center_distance(box_i, box_j)
                    if dist < config.collision_distance_threshold:
                        score = 1.0 - (dist / config.collision_distance_threshold)
                        max_score = max(max_score, score * 0.5)  # 距离分数权重较低

    return max_score


def _compute_intersection_score(tracks: Dict[int, Dict]) -> float:
    """
    计算轨迹交叉分数

    检测不同目标的轨迹线段是否相交。
    """
    track_ids = list(tracks.keys())
    if len(track_ids) < 2:
        return 0.0

    max_score = 0.0

    for i, tid1 in enumerate(track_ids):
        for tid2 in track_ids[i + 1:]:
            traj1 = tracks[tid1].get("trajectory", [])
            traj2 = tracks[tid2].get("trajectory", [])

            if len(traj1) < 2 or len(traj2) < 2:
                continue

            # 检测轨迹线段是否相交
            if _trajectories_intersect(traj1, traj2):
                max_score = 1.0
                break

        if max_score == 1.0:
            break

    return max_score


def _compute_deceleration_score(tracks: Dict[int, Dict], config: "TrajectoryScoreConfig") -> float:
    """
    计算急刹分数 - 检测速度突变

    遍历所有轨迹，计算速度变化梯度，检测急刹行为。
    """
    max_score = 0.0

    for track_id, track in tracks.items():
        traj = track.get("trajectory", [])
        if len(traj) < config.min_track_length:
            continue

        # 计算速度序列（基于bbox中心点位移）
        velocities = []
        for k in range(1, len(traj)):
            # traj格式: (ts, x1, y1, x2, y2)
            t0, x1_0, y1_0, x2_0, y2_0 = traj[k - 1]
            t1, x1_1, y1_1, x2_1, y2_1 = traj[k]

            # 中心点
            cx0 = (x1_0 + x2_0) / 2
            cy0 = (y1_0 + y2_0) / 2
            cx1 = (x1_1 + x2_1) / 2
            cy1 = (y1_1 + y2_1) / 2

            dt = t1 - t0
            if dt > 0:
                displacement = np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)
                v = displacement / dt
                velocities.append(v)

        if len(velocities) < 2:
            continue

        # 检测速度突降（急刹）
        for k in range(1, len(velocities)):
            v_prev = velocities[k - 1]
            v_curr = velocities[k]
            if v_prev > 10:  # 忽略几乎静止的目标（像素/秒）
                ratio = (v_prev - v_curr) / v_prev
                if ratio > config.deceleration_threshold:
                    score = min(1.0, ratio)
                    max_score = max(max_score, score)

    return max_score


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个bbox的IOU (Intersection over Union)

    Args:
        box1, box2: [x1, y1, x2, y2] 格式的边界框

    Returns:
        IOU值，0~1之间
    """
    if len(box1) < 4 or len(box2) < 4:
        return 0.0

    # 交集区域
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    # 并集区域
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _center_distance(box1: List[float], box2: List[float]) -> float:
    """
    计算两个bbox中心点的距离

    Args:
        box1, box2: [x1, y1, x2, y2] 格式的边界框

    Returns:
        欧氏距离（像素）
    """
    if len(box1) < 4 or len(box2) < 4:
        return float('inf')

    c1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    c2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _trajectories_intersect(traj1: List[Tuple], traj2: List[Tuple]) -> bool:
    """
    检测两条轨迹是否相交

    将轨迹简化为中心点连线，检测线段是否相交。

    Args:
        traj1, traj2: 轨迹列表，每项为 (ts, x1, y1, x2, y2)

    Returns:
        是否存在交叉
    """
    # 提取中心点序列
    def get_center_points(traj):
        points = []
        for t in traj:
            if len(t) >= 5:
                cx = (t[1] + t[3]) / 2
                cy = (t[2] + t[4]) / 2
                points.append((cx, cy))
        return points

    pts1 = get_center_points(traj1)
    pts2 = get_center_points(traj2)

    if len(pts1) < 2 or len(pts2) < 2:
        return False

    # 检测线段相交
    for i in range(len(pts1) - 1):
        for j in range(len(pts2) - 1):
            if _segments_intersect(pts1[i], pts1[i + 1], pts2[j], pts2[j + 1]):
                return True

    return False


def _segments_intersect(p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> bool:
    """
    检测线段p1-p2与p3-p4是否相交

    使用叉积法判断线段相交。

    Args:
        p1, p2: 第一条线段的端点
        p3, p4: 第二条线段的端点

    Returns:
        是否相交
    """
    def ccw(A, B, C):
        """Counter-clockwise test"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # 线段相交的充要条件
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
