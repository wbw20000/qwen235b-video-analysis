from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_roi(frame: "np.ndarray", roi_polygon: Optional[List[Tuple[int, int]]]) -> None:
    if roi_polygon:
        pts = np.array(roi_polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)


def save_raw_frames(
    frames: List[Tuple[float, "np.ndarray"]],
    save_dir: str,
    camera_id: str,
    date_str: str,
    clip_id: Optional[str] = None,
) -> List[Dict]:
    """
    保存原始帧（无YOLO叠加），用于渐进式VLM策略。

    Args:
        frames: [(timestamp, frame)]
        save_dir: 保存目录
        camera_id: 摄像头ID
        date_str: 日期字符串
        clip_id: clip标识符

    Returns:
        保存的帧信息列表 [{timestamp, path, clip_id}]
    """
    os.makedirs(save_dir, exist_ok=True)
    saved: List[Dict] = []

    for ts, frame in frames:
        if clip_id:
            fname = f"{camera_id}_{date_str}_{clip_id}_{ts:07.3f}_raw.jpg"
        else:
            fname = f"{camera_id}_{date_str}_{ts:07.3f}_raw.jpg"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, frame)
        saved.append({"timestamp": ts, "path": path, "clip_id": clip_id})

    return saved


def annotate_frames(
    frames: List[Tuple[float, "np.ndarray"]],
    frame_results: List[Dict],
    save_dir: str,
    camera_id: str,
    date_str: str,
    roi_polygon: Optional[List[Tuple[int, int]]] = None,
    clip_id: Optional[str] = None,
) -> List[Dict]:
    """
    在帧上绘制 bbox/track_id/ROI，保存文件。
    frames: [(timestamp, frame)]
    frame_results: [{"timestamp": ts, "detections": [...] }]
    clip_id: 可选的clip标识符，用于区分不同clip的标注帧，避免文件名冲突
    """
    os.makedirs(save_dir, exist_ok=True)
    annotated: List[Dict] = []

    # 诊断日志：检查长度一致性
    print(f"[VisualAnnotator] frames 长度: {len(frames)}, frame_results 长度: {len(frame_results)}")
    if len(frame_results) == 0:
        print(f"[VisualAnnotator] ⚠️ CRITICAL ERROR: frame_results为空！")
        print(f"[VisualAnnotator] ⚠️ 原因: YOLO检测失败或未启用")
        print(f"[VisualAnnotator] ⚠️ 影响: VLM将无法接收标注图片，所有视频会被判定为非事故")
        print(f"[VisualAnnotator] ⚠️ 解决: 检查config.detector.enabled配置")
        # 继续处理，但返回的标注列表会是空的
    elif len(frames) != len(frame_results):
        print(f"[VisualAnnotator] ⚠️ 长度不一致！后续帧可能没有检测框")

    for (ts, frame), res in zip(frames, frame_results):
        # 直接在原帧上绘制，避免内存复制（帧保存后会被释放）
        canvas = frame
        draw_roi(canvas, roi_polygon)
        for det in res.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            track_id = det.get("track_id", -1)
            score = det.get("score", 0.0)
            # 绘制检测框（加粗）
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # 绘制ID标签（增大字体，添加背景）
            label = f"ID:{track_id}"
            font_scale = 1.0  # 增大字体
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_y = max(text_h + 10, y1 - 5)
            # 黑色背景框
            cv2.rectangle(canvas, (x1, label_y - text_h - 5), (x1 + text_w + 5, label_y + 5), (0, 0, 0), -1)
            # 白色文字
            cv2.putText(canvas, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # 文件名加入clip_id以区分不同clip的标注帧
        if clip_id:
            fname = f"{camera_id}_{date_str}_{clip_id}_{ts:07.3f}_annotated.jpg"
        else:
            fname = f"{camera_id}_{date_str}_{ts:07.3f}_annotated.jpg"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, canvas)
        annotated.append({"timestamp": ts, "path": path, "clip_id": clip_id})
    return annotated
