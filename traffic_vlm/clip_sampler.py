from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Tuple

import cv2


def cut_clip_ffmpeg(input_path: str, start: float, end: float, output_path: str) -> bool:
    """使用 ffmpeg 裁剪片段，保留编码。"""
    duration = max(0.1, end - start)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
        "-avoid_negative_ts", "make_zero",  # 避免时间戳问题
        output_path,
    ]

    print(f"[clip_sampler] 剪辑视频: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
    print(f"[clip_sampler] 输出路径: {output_path}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=os.name == "nt",
            timeout=60  # 添加超时
        )

        # 验证输出文件是否存在且有效
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            output_size = os.path.getsize(output_path) / 1024  # KB
            print(f"[clip_sampler] ✓ 剪辑成功，文件大小: {output_size:.1f} KB")
            return True
        else:
            print(f"[clip_sampler] ✗ 剪辑失败：输出文件不存在或为空")
            return False

    except subprocess.TimeoutExpired:
        print(f"[clip_sampler] ✗ 剪辑超时")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[clip_sampler] ✗ FFmpeg 错误: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else '未知错误'}")
        return False
    except Exception as e:
        print(f"[clip_sampler] ✗ 剪辑异常: {str(e)}")
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
