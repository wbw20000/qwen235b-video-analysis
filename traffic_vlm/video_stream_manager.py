from __future__ import annotations

import cv2
import os
import subprocess
import sys
import tempfile
from typing import Generator, Optional, Tuple

from .config import StreamConfig


def check_nvenc_support() -> bool:
    """检查 FFmpeg 是否支持 NVIDIA NVENC 硬件加速"""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['ffmpeg', '-encoders'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5,
                                  shell=True)
        else:
            result = subprocess.run(['ffmpeg', '-encoders'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
        return 'h264_nvenc' in result.stdout
    except Exception:
        return False


def create_lowres_video_gpu(input_path: str, output_path: str,
                            width: int = 640, height: int = 360, fps: int = 12) -> bool:
    """
    使用 FFmpeg GPU 加速生成低分辨率视频。
    采用三级回退策略确保兼容性：
    1. 完整 GPU 流水线（GPU 解码 + GPU 缩放 + GPU 编码）
    2. GPU 解码 + CPU 缩放 + GPU 编码
    3. CPU 解码 + GPU 编码（最兼容）

    Args:
        input_path: 输入视频路径
        output_path: 输出低分辨率视频路径
        width: 目标宽度
        height: 目标高度
        fps: 目标帧率

    Returns:
        bool: 是否成功
    """
    # 确保分辨率是 2 的倍数（GPU 编码要求）
    width = (width // 2) * 2
    height = (height // 2) * 2

    print(f"[VideoStreamManager] 目标: {width}x{height}@{fps}fps", flush=True)

    strategies = [
        # 策略 1: 完整 GPU 流水线（最快）
        {
            'name': 'Full GPU Pipeline',
            'cmd': [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',  # 保持数据在 GPU 显存
                '-i', input_path,
                '-vf', f'scale_cuda={width}:{height}',  # GPU 缩放
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-r', str(fps),
                '-an',
                output_path
            ]
        },
        # 策略 2: GPU 解码 + CPU 缩放 + GPU 编码
        {
            'name': 'GPU Decode + CPU Scale',
            'cmd': [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',  # GPU 解码，但下载到 CPU
                '-i', input_path,
                '-vf', f'scale={width}:{height}',  # CPU 缩放
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-r', str(fps),
                '-an',
                output_path
            ]
        },
        # 策略 3: CPU 解码 + GPU 编码（最兼容）
        {
            'name': 'CPU Decode + GPU Encode',
            'cmd': [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', f'scale={width}:{height}',
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-r', str(fps),
                '-an',
                output_path
            ]
        }
    ]

    for strategy in strategies:
        try:
            print(f"[VideoStreamManager] 尝试: {strategy['name']}", flush=True)
            result = subprocess.run(
                strategy['cmd'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 遇到无法解码的字节用 � 替代，避免 GBK 解码崩溃
                timeout=300,
                shell=(sys.platform == 'win32')
            )

            if result.returncode == 0 and os.path.exists(output_path):
                print(f"[VideoStreamManager] [GPU] 成功: {strategy['name']}", flush=True)
                return True
            else:
                err_msg = result.stderr[:150] if result.stderr else 'Unknown error'
                print(f"[VideoStreamManager] {strategy['name']} 失败: {err_msg}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[VideoStreamManager] {strategy['name']} 超时", flush=True)
        except Exception as e:
            print(f"[VideoStreamManager] {strategy['name']} 异常: {e}", flush=True)

    print("[VideoStreamManager] 所有 GPU 策略均失败", flush=True)
    return False


class VideoStreamManager:
    """
    管理高/低分辨率双路流。
    - HighRes: 原始分辨率，支持随机访问抓帧
    - LowRes: 缩放 + 降帧率，连续读取用于运动检测（GPU 预处理加速）
    """

    def __init__(self, source: str, config: StreamConfig, camera_id: str = "camera-1"):
        self.source = source
        self.config = config
        self.camera_id = camera_id

        # 高清流
        self.high_cap = cv2.VideoCapture(source)
        if not self.high_cap.isOpened():
            raise ValueError(f"无法打开视频源: {source}")

        self.high_fps = self.high_cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_count = int(self.high_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.high_fps if self.high_fps else 0

        # 降帧比率：按 lowres_fps 近似丢帧（CPU 回退时使用）
        self.low_stride = max(int(self.high_fps / max(self.config.lowres_fps, 1)), 1)

        # 低清流：优先使用 GPU 预处理
        self.lowres_temp_path = None
        self.use_gpu_lowres = False
        self._create_lowres_stream()

    def _create_lowres_stream(self):
        """创建低分辨率流（优先 GPU 预处理，失败则回退 CPU）"""
        print(f"[VideoStreamManager] ========== 低分辨率流初始化 ==========", flush=True)
        print(f"[VideoStreamManager] 输入视频: {self.source}", flush=True)
        print(f"[VideoStreamManager] 目标分辨率: {self.config.lowres_size}, 帧率: {self.config.lowres_fps}", flush=True)

        # 检查是否支持 NVENC
        nvenc_supported = check_nvenc_support()
        print(f"[VideoStreamManager] NVENC 支持: {nvenc_supported}", flush=True)

        if nvenc_supported:
            # 创建临时文件路径
            temp_dir = tempfile.gettempdir()
            self.lowres_temp_path = os.path.join(
                temp_dir,
                f"lowres_{os.path.basename(self.source)}_{os.getpid()}.mp4"
            )

            # 尝试 GPU 预处理
            print(f"[VideoStreamManager] [GPU] 开始 FFmpeg GPU 预处理...", flush=True)
            success = create_lowres_video_gpu(
                self.source,
                self.lowres_temp_path,
                width=self.config.lowres_size[0],
                height=self.config.lowres_size[1],
                fps=self.config.lowres_fps
            )

            if success and os.path.exists(self.lowres_temp_path):
                self.low_cap = cv2.VideoCapture(self.lowres_temp_path)
                if self.low_cap.isOpened():
                    self.use_gpu_lowres = True
                    temp_size_mb = os.path.getsize(self.lowres_temp_path) / (1024 * 1024)
                    print(f"[VideoStreamManager] [GPU] 预处理完成，临时文件: {temp_size_mb:.2f} MB", flush=True)
                    print(f"[VideoStreamManager] ========== 使用 GPU 加速模式 ==========", flush=True)
                    return

        # GPU 失败或不支持，回退到原始方式
        print(f"[VideoStreamManager] [CPU] GPU 不可用或失败，回退到 CPU 实时缩放", flush=True)
        print(f"[VideoStreamManager] ========== 使用 CPU 模式（较慢）==========", flush=True)
        self.low_cap = cv2.VideoCapture(self.source)
        if not self.low_cap.isOpened():
            raise ValueError(f"无法打开低分辨率视频源: {self.source}")

    def iterate_lowres(self) -> Generator[Tuple[int, float, "cv2.Mat"], None, None]:
        """
        连续读取低分辨率帧，用于运动检测。
        Yields: (frame_idx, timestamp_seconds, frame)

        如果使用 GPU 预处理，直接读取预处理好的低分辨率视频（无需 resize）。
        如果回退到 CPU，则实时 resize。
        """
        idx = 0

        if self.use_gpu_lowres:
            # GPU 预处理模式：直接读取，帧已经是目标分辨率和帧率
            print(f"[VideoStreamManager] [GPU] 开始迭代低分辨率帧（GPU 预处理模式，无需 resize）", flush=True)
            lowres_fps = self.config.lowres_fps
            while True:
                ret, frame = self.low_cap.read()
                if not ret:
                    break
                timestamp = idx / lowres_fps if lowres_fps else 0.0
                yield idx, timestamp, frame  # 无需 resize
                idx += 1
            print(f"[VideoStreamManager] [GPU] 迭代完成，共 {idx} 帧", flush=True)
        else:
            # CPU 回退模式：实时 resize
            print(f"[VideoStreamManager] [CPU] 开始迭代低分辨率帧（CPU 实时 resize 模式）", flush=True)
            while True:
                ret, frame = self.low_cap.read()
                if not ret:
                    break

                timestamp = idx / self.high_fps if self.high_fps else 0.0

                if idx % self.low_stride == 0:
                    resized = cv2.resize(frame, self.config.lowres_size)
                    yield idx, timestamp, resized

                idx += 1
            print(f"[VideoStreamManager] [CPU] 迭代完成，共处理 {idx} 帧", flush=True)

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
        """释放资源并清理临时文件"""
        try:
            if self.high_cap:
                self.high_cap.release()
            if self.low_cap:
                self.low_cap.release()
        except Exception:
            pass

        # 清理 GPU 预处理的临时文件
        if self.lowres_temp_path and os.path.exists(self.lowres_temp_path):
            try:
                os.remove(self.lowres_temp_path)
                print(f"[VideoStreamManager] 已清理临时文件: {self.lowres_temp_path}", flush=True)
            except Exception as e:
                print(f"[VideoStreamManager] 清理临时文件失败: {e}", flush=True)


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
