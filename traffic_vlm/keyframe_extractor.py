from __future__ import annotations

import os
from typing import Dict, List
import cv2
from datetime import datetime, timedelta
from PIL import Image

from .config import StreamConfig
from .video_stream_manager import VideoStreamManager


def _timestamp_to_str(base_date: datetime, seconds: float) -> str:
    dt = base_date + timedelta(seconds=seconds)
    return dt.strftime("%H%M%S")


def extract_keyframes(
    manager: VideoStreamManager,
    triggers: List[Dict],
    save_dir: str,
    stream_config: StreamConfig,
    camera_id: str,
    base_date: datetime,
) -> List[Dict]:
    """
    根据触发时间戳抓取高清关键帧并保存。
    返回列表包含：timestamp, frame_idx, path, trigger_score, trigger_type
    """
    os.makedirs(save_dir, exist_ok=True)
    keyframes: List[Dict] = []
    last_ts = -1e9

    for trig in sorted(triggers, key=lambda x: x["timestamp"]):
        ts = float(trig["timestamp"])
        if ts - last_ts < stream_config.min_keyframe_interval:
            continue

        frame = manager.get_highres_frame_at(ts)
        if frame is None:
            continue

        frame_idx = int(ts * manager.high_fps) if manager.high_fps else 0
        time_str = _timestamp_to_str(base_date, ts)
        filename = f"{camera_id}_{base_date.strftime('%Y%m%d')}_{time_str}_{frame_idx:06d}_keyframe.jpg"
        path = os.path.join(save_dir, filename)

        cv2.imwrite(path, frame)

        # 同时保留内存图像（避免写盘后再读回）
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        keyframes.append(
            {
                "camera_id": camera_id,
                "timestamp": ts,
                "frame_idx": frame_idx,
                "path": path,
                "image": pil_image,  # 内存图像，供embedding直接使用
                "trigger_score": trig.get("score", 0.0),
                "trigger_type": trig.get("type", "unknown"),
            }
        )
        last_ts = ts

    return keyframes
