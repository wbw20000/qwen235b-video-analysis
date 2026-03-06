#!/usr/bin/env python3
"""
PreProcessor - 共享预处理服务

功能:
1. 消费 video_tasks 队列
2. FFmpeg NVDEC + CUDA MOG2 运动检测提取关键帧
3. SigLIP 计算 Embedding (内置，不调用外部服务)
4. 缓存到 Redis (TTL 5 分钟)
5. 发布到 4 个下游队列 (accident_tasks, mv_tasks, ebike_tasks, ads_tasks)

设计目标:
- 消除 4 个分析器重复的 MOG2 + Embedding 计算
- HPA 自动伸缩，保证实时性
- 每个 Pod ~1.3GB 显存 (MOG2 ~100MB + SigLIP ~1.2GB)
"""
import os
import sys
import time
import signal
import json
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
from PIL import Image

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, VideoTask
from workers.common.cache_client import (
    CacheClient, PreprocessedResult, FrameCache, DownstreamTask
)


# ========== 配置 ==========
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")
MAX_TASKS_BEFORE_EXIT = int(os.getenv("MAX_TASKS", "100"))

# CUDA 设备配置
CUDA_DEVICE = os.getenv("CUDA_DEVICE", os.getenv("FFMPEG_NVDEC_DEVICE", "0"))

# MOG2 运动检测配置
MOG2_FG_RATIO_THRESHOLD = float(os.getenv("MOG2_FG_RATIO_THRESHOLD", "0.015"))
MOG2_LOWRES_WIDTH = int(os.getenv("MOG2_LOWRES_WIDTH", "640"))
MOG2_LOWRES_HEIGHT = int(os.getenv("MOG2_LOWRES_HEIGHT", "360"))
MOG2_LOWRES_FPS = float(os.getenv("MOG2_LOWRES_FPS", "12"))
MOG2_DEBOUNCE_FRAMES = int(os.getenv("MOG2_DEBOUNCE_FRAMES", "3"))
MOG2_ALWAYS_SAMPLE_INTERVAL = float(os.getenv("MOG2_ALWAYS_SAMPLE_INTERVAL", "5.0"))

# 评测模式: 绕过运动检测过滤，均匀采样帧
BYPASS_MOTION_FILTER = os.getenv("BYPASS_MOTION_FILTER", "").lower() in ("1", "true", "yes")
BYPASS_SAMPLE_INTERVAL = float(os.getenv("BYPASS_SAMPLE_INTERVAL", "1.0"))  # 每秒采样1帧

# SigLIP 模型配置
SIGLIP_MODEL_PATH = os.getenv("SIGLIP_MODEL_PATH", "/data/models/siglip-base-patch16-384")

# 图片压缩配置
COMPRESS_MAX_SIZE = int(os.getenv("COMPRESS_MAX_SIZE", "640"))
COMPRESS_QUALITY = int(os.getenv("COMPRESS_QUALITY", "85"))

# 缓存配置
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # 5 分钟

# 帧存储目录 (共享 PVC, Analyzer 可访问)
FRAMES_DIR = os.getenv("FRAMES_DIR", "/data1/frames")

# CUDA MOG2 可用性
_CUDA_MOG2_AVAILABLE = None


def _check_cuda_mog2():
    """检测 cv2.cuda MOG2 是否可用"""
    global _CUDA_MOG2_AVAILABLE
    if _CUDA_MOG2_AVAILABLE is True:
        return True
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            cv2.cuda.setDevice(int(CUDA_DEVICE))
            test_mog2 = cv2.cuda.createBackgroundSubtractorMOG2(
                history=10, varThreshold=16, detectShadows=False
            )
            test_frame = cv2.cuda_GpuMat(np.zeros((64, 64), dtype=np.uint8))
            test_stream = cv2.cuda.Stream()
            test_mog2.apply(test_frame, -1, test_stream)
            _CUDA_MOG2_AVAILABLE = True
        else:
            _CUDA_MOG2_AVAILABLE = False
    except Exception as e:
        logging.getLogger("preprocessor").warning(f"CUDA MOG2 检测失败: {e}")
        _CUDA_MOG2_AVAILABLE = None
        return False
    return _CUDA_MOG2_AVAILABLE


@dataclass
class FrameInfo:
    """帧信息 (内部使用)"""
    frame_idx: int
    timestamp_sec: float
    frame_path: str
    embedding: Optional[np.ndarray] = None


class PreProcessor:
    """共享预处理服务"""

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        results_dir: str = RESULTS_DIR,
        siglip_model_path: str = SIGLIP_MODEL_PATH
    ):
        self.redis = RedisStreamClient(redis_host, redis_port)
        self.cache = CacheClient(redis_host, redis_port)
        self.consumer_group = "preprocessors"
        self.consumer_name = os.getenv("HOSTNAME", f"preprocessor_{os.getpid()}")

        self.results_dir = Path(results_dir) / "preprocessor"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger("preprocessor")
        self.log = LogContext(self.logger, stage="preprocessor")

        self.task_count = 0
        self.running = True

        # SigLIP 模型
        self.siglip_model = None
        self.siglip_processor = None
        self.device = None
        self.siglip_model_path = siglip_model_path

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.log.info(f"收到信号 {signum}，准备优雅退出")
        self.running = False

    def cleanup(self):
        """清理资源（删除消费者注册）"""
        try:
            pending = self.redis.delete_video_consumer(
                self.consumer_group, self.consumer_name
            )
            self.log.info(f"已清理消费者 {self.consumer_name}，释放 {pending} 个 pending 消息")
        except Exception as e:
            self.log.warning(f"清理消费者失败: {e}")

    def load_siglip_model(self):
        """加载 SigLIP 模型"""
        try:
            from transformers import SiglipModel, SiglipProcessor

            self.device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

            self.log.info(f"加载 SigLIP 模型: {self.siglip_model_path} -> {self.device}")

            self.siglip_processor = SiglipProcessor.from_pretrained(self.siglip_model_path)
            self.siglip_model = SiglipModel.from_pretrained(self.siglip_model_path)
            self.siglip_model.to(self.device)
            self.siglip_model.eval()

            self.log.info("SigLIP 模型加载完成")

        except Exception as e:
            self.log.error(f"加载 SigLIP 模型失败: {e}")
            raise

    def _get_video_info(self, video_path: str) -> Tuple[float, int, int, int]:
        """获取视频信息 (fps, width, height, total_frames)"""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-show_format",
                str(video_path)
            ]
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            info = json.loads(result.stdout)
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    fps_str = stream.get("r_frame_rate", "25/1")
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den) if float(den) > 0 else 25.0
                    width = int(stream.get("width", 1920))
                    height = int(stream.get("height", 1080))
                    nb_frames = int(stream.get("nb_frames", 0))
                    if nb_frames == 0:
                        duration = float(info.get("format", {}).get("duration", 60))
                        nb_frames = int(duration * fps)
                    return fps, width, height, nb_frames
        except Exception:
            pass
        return 25.0, 1920, 1080, 0

    def extract_frames_with_mog2(
        self,
        video_path: str,
        job_id: str = "",
        trace_id: str = ""
    ) -> Tuple[List[FrameInfo], str]:
        """FFmpeg NVDEC + CUDA MOG2 运动检测

        Returns:
            (frames, tmpdir_path)
        """
        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            self.log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames, ""

        # 使用共享 PVC 目录存储帧，Analyzer 可访问
        frame_subdir = job_id if job_id else f"tmp_{int(time.time()*1000)}"
        tmpdir = Path(FRAMES_DIR) / frame_subdir
        tmpdir.mkdir(parents=True, exist_ok=True)
        proc = None

        try:
            orig_fps, orig_width, orig_height, total_frames = self._get_video_info(str(video_path))

            use_cuda_mog2 = _check_cuda_mog2()
            decode_label = f"NVDEC:{CUDA_DEVICE}+{'CUDA' if use_cuda_mog2 else 'CPU'}MOG2"
            self.log.info(
                f"MOG2 运动检测: {orig_width}x{orig_height}@{orig_fps:.1f}fps -> "
                f"{MOG2_LOWRES_WIDTH}x{MOG2_LOWRES_HEIGHT}@{MOG2_LOWRES_FPS}fps ({decode_label})",
                job_id=job_id, trace_id=trace_id
            )

            # 阶段1: FFmpeg NVDEC 低分辨率管道 + MOG2
            t0 = time.time()

            # 首先尝试 NVDEC 硬件解码
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_device", CUDA_DEVICE,
                "-i", str(video_path),
                "-vf", f"scale={MOG2_LOWRES_WIDTH}:{MOG2_LOWRES_HEIGHT},fps={MOG2_LOWRES_FPS}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-v", "error", "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            frame_size = MOG2_LOWRES_WIDTH * MOG2_LOWRES_HEIGHT * 3

            # 尝试初始化 CUDA MOG2，失败则回退到 CPU
            actual_cuda_mog2 = False
            bg_subtractor = None
            cuda_stream = None
            gpu_frame = None

            if use_cuda_mog2:
                try:
                    cv2.cuda.setDevice(int(CUDA_DEVICE))
                    bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                        history=400, varThreshold=16, detectShadows=True
                    )
                    cuda_stream = cv2.cuda.Stream()
                    gpu_frame = cv2.cuda_GpuMat()
                    actual_cuda_mog2 = True
                except Exception as cuda_err:
                    self.log.warning(f"CUDA MOG2 初始化失败，回退到 CPU: {cuda_err}", job_id=job_id)
                    actual_cuda_mog2 = False

            if not actual_cuda_mog2:
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=400, varThreshold=16, detectShadows=True
                )

            trigger_frame_indices = []
            frame_idx = 0
            motion_streak = 0
            last_force_ts = -1e9
            cuda_fallback_done = False

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    MOG2_LOWRES_HEIGHT, MOG2_LOWRES_WIDTH, 3
                )
                timestamp = frame_idx / MOG2_LOWRES_FPS

                try:
                    if actual_cuda_mog2:
                        gpu_frame.upload(frame)
                        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                        gpu_fg = bg_subtractor.apply(gpu_gray, -1, cuda_stream)
                        fg_mask = gpu_fg.download()
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        fg_mask = bg_subtractor.apply(gray)
                        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                except Exception as mog2_err:
                    # CUDA MOG2 运行时失败，回退到 CPU
                    if actual_cuda_mog2 and not cuda_fallback_done:
                        self.log.warning(f"CUDA MOG2 运行失败，回退到 CPU: {mog2_err}", job_id=job_id)
                        actual_cuda_mog2 = False
                        cuda_fallback_done = True
                        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                            history=400, varThreshold=16, detectShadows=True
                        )
                        # 重新处理当前帧
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        gray = cv2.GaussianBlur(gray, (3, 3), 0)
                        fg_mask = bg_subtractor.apply(gray)
                        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                    else:
                        raise

                fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size

                if fg_ratio >= MOG2_FG_RATIO_THRESHOLD:
                    motion_streak += 1
                else:
                    motion_streak = 0

                triggered = False
                if motion_streak >= MOG2_DEBOUNCE_FRAMES:
                    triggered = True
                    motion_streak = 0
                if timestamp - last_force_ts >= MOG2_ALWAYS_SAMPLE_INTERVAL:
                    triggered = True
                    last_force_ts = timestamp

                if triggered:
                    orig_frame_idx = int(timestamp * orig_fps)
                    trigger_frame_indices.append((orig_frame_idx, timestamp))

                frame_idx += 1

            # 获取 FFmpeg stderr
            _, ffmpeg_stderr = proc.communicate()
            t1 = time.time()

            # 如果 NVDEC 失败（0 帧），尝试软件解码 + CPU MOG2
            if frame_idx == 0 and ffmpeg_stderr:
                stderr_text = ffmpeg_stderr.decode('utf-8', errors='ignore').strip()
                if stderr_text:
                    self.log.warning(f"NVDEC 失败: {stderr_text[:200]}", job_id=job_id)

                # 回退到软件解码 + CPU MOG2
                self.log.info("回退到软件解码 + CPU MOG2", job_id=job_id, trace_id=trace_id)
                cmd_sw = [
                    "ffmpeg",
                    "-i", str(video_path),
                    "-vf", f"scale={MOG2_LOWRES_WIDTH}:{MOG2_LOWRES_HEIGHT},fps={MOG2_LOWRES_FPS}",
                    "-f", "rawvideo", "-pix_fmt", "bgr24",
                    "-v", "error", "-"
                ]
                proc = subprocess.Popen(cmd_sw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # 强制使用 CPU MOG2（避免 GPU 问题）
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=400, varThreshold=16, detectShadows=True
                )
                motion_streak = 0
                last_force_ts = -1e9

                # 重新处理帧（纯 CPU）
                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        MOG2_LOWRES_HEIGHT, MOG2_LOWRES_WIDTH, 3
                    )
                    timestamp = frame_idx / MOG2_LOWRES_FPS

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    fg_mask = bg_subtractor.apply(gray)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

                    fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size

                    if fg_ratio >= MOG2_FG_RATIO_THRESHOLD:
                        motion_streak += 1
                    else:
                        motion_streak = 0

                    triggered = False
                    if motion_streak >= MOG2_DEBOUNCE_FRAMES:
                        triggered = True
                        motion_streak = 0
                    if timestamp - last_force_ts >= MOG2_ALWAYS_SAMPLE_INTERVAL:
                        triggered = True
                        last_force_ts = timestamp

                    if triggered:
                        orig_frame_idx = int(timestamp * orig_fps)
                        trigger_frame_indices.append((orig_frame_idx, timestamp))

                    frame_idx += 1

                proc.communicate()
                t1 = time.time()

            self.log.info(
                f"阶段1 MOG2: {frame_idx} 帧 -> {len(trigger_frame_indices)} 触发点, "
                f"耗时={t1-t0:.1f}s",
                job_id=job_id, trace_id=trace_id
            )

            if not trigger_frame_indices:
                if BYPASS_MOTION_FILTER:
                    # 评测模式: 绕过运动检测，均匀采样
                    self.log.info(
                        f"BYPASS_MOTION_FILTER: 无触发点但强制采样 (间隔={BYPASS_SAMPLE_INTERVAL}s)",
                        job_id=job_id
                    )
                    # 使用均匀采样替代触发点
                    for t in np.arange(0, duration, BYPASS_SAMPLE_INTERVAL):
                        orig_idx = int(t * orig_fps)
                        trigger_frame_indices.append((orig_idx, float(t)))
                    # 限制最大帧数
                    if len(trigger_frame_indices) > 30:
                        step = len(trigger_frame_indices) // 30
                        trigger_frame_indices = trigger_frame_indices[::step][:30]
                else:
                    self.log.warning("MOG2 无触发点，跳过", job_id=job_id)
                    return frames, str(tmpdir)

            # 阶段2: FFmpeg NVDEC 仅提取触发帧
            t2 = time.time()

            select_parts = [f"eq(n\\,{idx})" for idx, _ in trigger_frame_indices]
            select_expr = "+".join(select_parts)

            output_pattern = str(tmpdir / "frame_%04d.jpg")
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_device", CUDA_DEVICE,
                "-i", str(video_path),
                "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                "-vsync", "0",
                "-q:v", "2",
                output_pattern
            ]
            subprocess.run(cmd, capture_output=True, timeout=120)

            t3 = time.time()

            for i, (orig_idx, timestamp) in enumerate(trigger_frame_indices):
                frame_path = str(tmpdir / f"frame_{i+1:04d}.jpg")
                if Path(frame_path).exists():
                    frames.append(FrameInfo(
                        frame_idx=i,
                        timestamp_sec=timestamp,
                        frame_path=frame_path
                    ))

            total_time = t3 - t0
            filter_pct = 100 * (1 - len(frames) / max(1, frame_idx))
            self.log.info(
                f"MOG2 提取 {len(frames)} 帧 (过滤 {filter_pct:.1f}% 静止帧), "
                f"阶段1={t1-t0:.1f}s 阶段2={t3-t2:.1f}s 总计={total_time:.1f}s",
                job_id=job_id, trace_id=trace_id
            )

        except Exception as e:
            self.log.error(f"MOG2 抽帧异常: {e}", job_id=job_id, trace_id=trace_id)
            import traceback
            self.log.error(traceback.format_exc())
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

        return frames, str(tmpdir)

    def compress_image(self, image_path: str, max_size: int = COMPRESS_MAX_SIZE) -> Image.Image:
        """压缩图片并返回 PIL Image"""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
        return img

    def compute_embeddings(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """使用内置 SigLIP 计算帧嵌入"""
        if not frames or self.siglip_model is None:
            return frames

        t0 = time.time()

        # 批量加载图片
        images = []
        valid_indices = []
        for i, frame in enumerate(frames):
            try:
                img = self.compress_image(frame.frame_path)
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                self.log.warning(f"读取帧失败: {frame.frame_path}, {e}")

        if not images:
            return frames

        try:
            # 批量处理
            inputs = self.siglip_processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.siglip_model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()

            # 归一化
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            for i, idx in enumerate(valid_indices):
                frames[idx].embedding = embeddings[i]

            t1 = time.time()
            self.log.info(
                f"SigLIP 计算 {len(embeddings)} 帧嵌入, 耗时={t1-t0:.2f}s",
                job_id=job_id, trace_id=trace_id
            )

        except Exception as e:
            self.log.error(f"计算嵌入失败: {e}", job_id=job_id, trace_id=trace_id)

        return frames

    def process_task(self, task: VideoTask) -> bool:
        """处理单个任务"""
        job_id = task.job_id
        trace_id = task.trace_id
        t_start = time.time()

        self.log.info(f"开始处理: {task.window_path}", job_id=job_id, trace_id=trace_id)

        # 1. MOG2 运动检测提取关键帧
        frames, tmpdir = self.extract_frames_with_mog2(
            task.window_path, job_id=job_id, trace_id=trace_id
        )

        if not frames:
            self.log.warning("无关键帧，跳过", job_id=job_id, trace_id=trace_id)
            # 清理临时目录
            if tmpdir:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            # 标记任务为完成（无帧）
            self.redis.set_task_status(job_id, "done", self.consumer_name)
            return True

        # 2. 计算 Embedding
        frames = self.compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

        # 3. 构建缓存数据
        frame_caches = []
        for f in frames:
            if f.embedding is not None:
                frame_caches.append(FrameCache(
                    idx=f.frame_idx,
                    timestamp_sec=f.timestamp_sec,
                    path=f.frame_path,
                    embedding=f.embedding.tolist()
                ))

        if not frame_caches:
            self.log.warning("无有效嵌入，跳过", job_id=job_id, trace_id=trace_id)
            return True

        result = PreprocessedResult(
            job_id=job_id,
            camera_id=task.camera_id,
            video_path=task.window_path,
            created_at=time.time(),
            trace_id=trace_id,
            frames=frame_caches
        )

        # 4. 缓存到 Redis
        cache_key = job_id
        self.cache.set(cache_key, result, ttl=CACHE_TTL)

        # 5. 发布到 4 个下游队列
        downstream_task = DownstreamTask(
            job_id=job_id,
            camera_id=task.camera_id,
            cache_key=cache_key,
            trace_id=trace_id,
            video_path=task.window_path
        )
        msg_ids = self.cache.publish_downstream_task(downstream_task)

        t_end = time.time()
        self.log.info(
            f"完成: {len(frame_caches)} 帧缓存, 发布到 {len(msg_ids)} 队列, "
            f"耗时={t_end-t_start:.1f}s",
            job_id=job_id, trace_id=trace_id
        )

        return True

    def run(self):
        """主循环"""
        self.log.info(f"PreProcessor 启动: consumer={self.consumer_name}, GPU={CUDA_DEVICE}")

        # 加载 SigLIP 模型
        self.load_siglip_model()

        while self.running:
            try:
                # 从 video_tasks 队列读取任务
                tasks = self.redis.read_video_tasks(
                    consumer_group=self.consumer_group,
                    consumer_name=self.consumer_name,
                    count=1,
                    block_ms=5000
                )

                if not tasks:
                    continue

                for msg_id, task in tasks:
                    try:
                        success = self.process_task(task)
                        if success:
                            self.redis.ack_video_task(self.consumer_group, msg_id)
                            self.task_count += 1
                        else:
                            self.log.warning(f"任务处理失败: {task.job_id}")
                    except Exception as e:
                        self.log.error(f"任务异常: {task.job_id}, {e}")
                        import traceback
                        self.log.error(traceback.format_exc())
                        self.redis.add_to_dlq(task, str(e))
                        self.redis.ack_video_task(self.consumer_group, msg_id)

                # 任务计数检查
                if self.task_count >= MAX_TASKS_BEFORE_EXIT:
                    self.log.info(f"达到最大任务数 {MAX_TASKS_BEFORE_EXIT}，退出重启")
                    break

            except Exception as e:
                self.log.error(f"主循环异常: {e}")
                import traceback
                self.log.error(traceback.format_exc())
                time.sleep(5)

        # 清理消费者注册
        self.cleanup()
        self.log.info(f"PreProcessor 退出，共处理 {self.task_count} 任务")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PreProcessor - 共享预处理服务")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    parser.add_argument("--siglip-model", default=os.getenv("SIGLIP_MODEL_PATH", "/data/models/siglip-base-patch16-384"))
    args = parser.parse_args()

    preprocessor = PreProcessor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        results_dir=args.results_dir,
        siglip_model_path=args.siglip_model
    )
    preprocessor.run()


if __name__ == "__main__":
    main()
