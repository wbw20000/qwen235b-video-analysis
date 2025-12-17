from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_roi(frame: "np.ndarray", roi_polygon: Optional[List[Tuple[int, int]]]) -> None:
    if roi_polygon:
        pts = np.array(roi_polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2)


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
    if len(frames) != len(frame_results):
        print(f"[VisualAnnotator] ⚠️ 长度不一致！后续帧可能没有检测框")

    for (ts, frame), res in zip(frames, frame_results):
        canvas = frame.copy()
        draw_roi(canvas, roi_polygon)
        for det in res.get("detections", []):
            x1, y1, x2, y2 = map(int, det["bbox"])
            track_id = det.get("track_id", -1)
            score = det.get("score", 0.0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {track_id} {score:.2f}"
            cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 文件名加入clip_id以区分不同clip的标注帧
        if clip_id:
            fname = f"{camera_id}_{date_str}_{clip_id}_{ts:07.3f}_annotated.jpg"
        else:
            fname = f"{camera_id}_{date_str}_{ts:07.3f}_annotated.jpg"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, canvas)
        annotated.append({"timestamp": ts, "path": path, "clip_id": clip_id})
    return annotated
