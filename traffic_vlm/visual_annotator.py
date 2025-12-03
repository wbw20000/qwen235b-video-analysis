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
) -> List[Dict]:
    """
    在帧上绘制 bbox/track_id/ROI，保存文件。
    frames: [(timestamp, frame)]
    frame_results: [{"timestamp": ts, "detections": [...] }]
    """
    os.makedirs(save_dir, exist_ok=True)
    annotated: List[Dict] = []

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

        fname = f"{camera_id}_{date_str}_{ts:07.3f}_annotated.jpg"
        path = os.path.join(save_dir, fname)
        cv2.imwrite(path, canvas)
        annotated.append({"timestamp": ts, "path": path})
    return annotated
