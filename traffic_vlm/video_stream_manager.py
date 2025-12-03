from __future__ import annotations

import cv2
import os
from typing import Generator, Optional, Tuple

from .config import StreamConfig


class VideoStreamManager:
    """
    管理高/低分辨率双路流。
    - HighRes: 原始分辨率，支持随机访问抓帧
    - LowRes: 缩放 + 降帧率，连续读取用于运动检测
    """

    def __init__(self, source: str, config: StreamConfig, camera_id: str = "camera-1"):
        self.source = source
        self.config = config
        self.camera_id = camera_id

        self.high_cap = cv2.VideoCapture(source)
        if not self.high_cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        self.low_cap = cv2.VideoCapture(source)
        if not self.low_cap.isOpened():
            raise ValueError(f"无法打开低分辨率视频源: {source}")

        self.high_fps = self.high_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.high_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.high_fps if self.high_fps else 0

        # 降帧比率：按 lowres_fps 近似丢帧
        self.low_stride = max(int(self.high_fps / max(self.config.lowres_fps, 1)), 1)

    def iterate_lowres(self) -> Generator[Tuple[int, float, "cv2.Mat"], None, None]:
        """
        连续读取低分辨率帧，用于运动检测。
        Yields: (frame_idx, timestamp_seconds, frame)
        """
        idx = 0
        while True:
            ret, frame = self.low_cap.read()
            if not ret:
                break

            timestamp = idx / self.high_fps if self.high_fps else 0.0

            if idx % self.low_stride == 0:
                resized = cv2.resize(frame, self.config.lowres_size)
                yield idx, timestamp, resized

            idx += 1

    def get_highres_frame_at(self, timestamp: float) -> Optional["cv2.Mat"]:
        """根据时间戳抓取最接近的高清帧。"""
        if self.high_fps <= 0:
            return None
        frame_idx = int(timestamp * self.high_fps)
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self.high_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.high_cap.read()
        return frame if ret else None

    def get_highres_frame_by_index(self, frame_idx: int) -> Optional["cv2.Mat"]:
        """根据帧号抓取高清帧。"""
        frame_idx = max(0, min(frame_idx, self.frame_count - 1))
        self.high_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.high_cap.read()
        return frame if ret else None

    def release(self):
        try:
            if self.high_cap:
                self.high_cap.release()
            if self.low_cap:
                self.low_cap.release()
        except Exception:
            pass


def build_output_paths(base_dir: str, camera_id: str, date_str: str) -> dict:
    """创建数据输出目录并返回路径字典。"""
    root = os.path.join(base_dir, camera_id, date_str)
    paths = {
        "root": root,
        "lowres_debug": os.path.join(root, "lowres_debug_frames"),
        "keyframes": os.path.join(root, "keyframes"),
        "raw_suspect_clips": os.path.join(root, "raw_suspect_clips"),
        "refined_clips": os.path.join(root, "refined_clips"),
        "annotated_frames": os.path.join(root, "annotated_frames"),
        "logs": os.path.join(root, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths
