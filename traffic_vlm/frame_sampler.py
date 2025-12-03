from __future__ import annotations

import os
from typing import List, Tuple

import cv2


def sample_uniform(video_path: str, fps: float = 1.0, max_frames: int = 0) -> List[Tuple[float, "cv2.Mat"]]:
    """
    简单均匀采样（备用/调试）。
    返回 (timestamp, frame) 列表。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(int(video_fps / max(fps, 0.1)), 1)

    frames = []
    idx = 0
    extracted = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            ts = idx / video_fps
            frames.append((ts, frame))
            extracted += 1
            if max_frames and extracted >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def save_frames(frames: List[Tuple[float, "cv2.Mat"]], save_dir: str, prefix: str = "frame") -> List[str]:
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i, (ts, frame) in enumerate(frames):
        path = os.path.join(save_dir, f"{prefix}_{i:04d}_{ts:07.3f}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths
