"""
ROI工具模块

提供边界框操作和ROI裁剪功能，用于VLM高精度事故分析。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def expand_box(
    box: Tuple[int, int, int, int],
    scale: float = 1.5,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    扩展边界框

    Args:
        box: (x1, y1, x2, y2) 边界框
        scale: 扩展比例（1.5表示扩大50%）
        image_size: (width, height) 图像尺寸，用于边界裁剪

    Returns:
        扩展后的边界框 (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_w = w * scale
    new_h = h * scale

    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)

    if image_size:
        new_x1, new_y1, new_x2, new_y2 = clamp_to_image(
            (new_x1, new_y1, new_x2, new_y2), image_size
        )

    return (new_x1, new_y1, new_x2, new_y2)


def union_box(
    boxes: List[Tuple[int, int, int, int]],
    scale: float = 1.0,
    image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, int, int]:
    """
    计算多个边界框的并集

    Args:
        boxes: 边界框列表 [(x1, y1, x2, y2), ...]
        scale: 并集后的扩展比例
        image_size: 图像尺寸，用于边界裁剪

    Returns:
        并集边界框 (x1, y1, x2, y2)
    """
    if not boxes:
        raise ValueError("boxes列表不能为空")

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    union = (x1, y1, x2, y2)

    if scale != 1.0:
        union = expand_box(union, scale, image_size)
    elif image_size:
        union = clamp_to_image(union, image_size)

    return union


def clamp_to_image(
    box: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """
    将边界框裁剪到图像边界内

    Args:
        box: (x1, y1, x2, y2) 边界框
        image_size: (width, height) 图像尺寸

    Returns:
        裁剪后的边界框 (x1, y1, x2, y2)
    """
    w, h = image_size
    x1 = max(0, min(box[0], w - 1))
    y1 = max(0, min(box[1], h - 1))
    x2 = max(x1 + 1, min(box[2], w))
    y2 = max(y1 + 1, min(box[3], h))
    return (x1, y1, x2, y2)


def crop_roi(
    image: "np.ndarray",
    box: Tuple[int, int, int, int],
    min_size: int = 64,
) -> "np.ndarray":
    """
    从图像中裁剪ROI区域

    Args:
        image: 输入图像 (H, W, C)
        box: (x1, y1, x2, y2) 边界框
        min_size: 最小输出尺寸

    Returns:
        裁剪后的ROI图像
    """
    if cv2 is None:
        raise ImportError("cv2 not available")

    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_to_image(box, (w, h))

    roi = image[y1:y2, x1:x2].copy()

    # 确保最小尺寸
    if roi.shape[0] < min_size or roi.shape[1] < min_size:
        scale = max(min_size / roi.shape[0], min_size / roi.shape[1])
        new_w = int(roi.shape[1] * scale)
        new_h = int(roi.shape[0] * scale)
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return roi


def create_event_window_roi(
    image: "np.ndarray",
    center: Optional[Tuple[int, int]] = None,
    window_size: int = 640,
) -> Tuple["np.ndarray", Tuple[int, int, int, int]]:
    """
    创建以指定中心点为中心的固定窗口ROI

    Args:
        image: 输入图像
        center: (cx, cy) 中心点，None则使用图像中心
        window_size: 窗口大小

    Returns:
        (roi_image, box): ROI图像和对应的边界框
    """
    h, w = image.shape[:2]

    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    half = window_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    box = clamp_to_image((x1, y1, x2, y2), (w, h))
    roi = crop_roi(image, box)

    return roi, box


def extract_participant_boxes(
    frame_result: Dict,
    track_ids: Optional[List[int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    从检测结果中提取参与者的边界框

    Args:
        frame_result: 单帧检测结果，包含 detections 列表
        track_ids: 指定的track_id列表，None则提取所有

    Returns:
        边界框列表
    """
    boxes = []
    detections = frame_result.get("detections", [])

    for det in detections:
        tid = det.get("track_id")
        if track_ids is not None and tid not in track_ids:
            continue

        bbox = det.get("bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            boxes.append((x1, y1, x2, y2))

    return boxes


def compute_union_roi_from_detections(
    frame_results: List[Dict],
    scale: float = 1.5,
    image_size: Optional[Tuple[int, int]] = None,
    track_ids: Optional[List[int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    从多帧检测结果中计算Union ROI

    聚合所有帧中指定目标的边界框，计算并集ROI。

    Args:
        frame_results: 帧检测结果列表
        scale: ROI扩展比例
        image_size: 图像尺寸
        track_ids: 指定的track_id列表

    Returns:
        Union ROI边界框，无检测时返回None
    """
    all_boxes = []

    for fr in frame_results:
        boxes = extract_participant_boxes(fr, track_ids)
        all_boxes.extend(boxes)

    if not all_boxes:
        return None

    return union_box(all_boxes, scale, image_size)


def find_risk_peak_center(
    frame_results: List[Dict],
    tracks: Dict,
) -> Optional[Tuple[int, int]]:
    """
    找到风险峰值时刻的中心位置

    用于event_window模式下确定ROI中心。

    Args:
        frame_results: 帧检测结果列表
        tracks: 轨迹字典

    Returns:
        (cx, cy) 中心坐标，无法确定时返回None
    """
    if not frame_results:
        return None

    # 找到检测数量最多的帧（通常是事故发生时刻）
    best_frame = max(frame_results, key=lambda f: len(f.get("detections", [])))
    boxes = extract_participant_boxes(best_frame)

    if not boxes:
        return None

    # 计算所有框的中心
    centers = [((b[0] + b[2]) // 2, (b[1] + b[3]) // 2) for b in boxes]
    cx = sum(c[0] for c in centers) // len(centers)
    cy = sum(c[1] for c in centers) // len(centers)

    return (cx, cy)
