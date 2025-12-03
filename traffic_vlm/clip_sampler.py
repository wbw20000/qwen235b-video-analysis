from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Tuple

import cv2


def cut_clip_ffmpeg(input_path: str, start: float, end: float, output_path: str) -> bool:
    """使用 ffmpeg 裁剪片段，保留编码。"""
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=os.name == "nt")
        return True
    except Exception:
        return False


def sample_frames_from_clip(
    video_path: str, start: float, end: float, num_frames: int
) -> List[Tuple[float, "cv2.Mat"]]:
    """
    从视频区间均匀采样若干帧。
    返回 [(timestamp, frame_bgr)]。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = max(0.1, end - start)
    positions = [start + i * duration / max(1, num_frames - 1) for i in range(num_frames)]

    samples = []
    for ts in positions:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            samples.append((ts, frame))
    cap.release()
    return samples
