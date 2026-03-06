#!/usr/bin/env python3
"""
Semantic Analyzer 基类 - 多语义检测共享逻辑
支持: accident, mv_violation, ebike_violation, ads_behavior

设计原则:
- FFmpeg NVDEC GPU 解码 + CUDA MOG2 运动检测 (单低分辨率遍 + 触发帧提取)
- 帧压缩 (640px + JPEG 85) 降低 Embedding/VLM 传输开销
- 每个分析类型独立模板和 prompt
"""
import os
import sys
import time
import signal
import json
import base64
import gzip
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import tempfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import requests
import numpy as np

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, VideoTask, ResultTask


# 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080")
VLM_PROXY_URL = os.getenv("VLM_PROXY_URL", "http://localhost:8001")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")
MAX_TASKS_BEFORE_EXIT = int(os.getenv("MAX_TASKS", "50"))
CLIP_SCORE_THRESHOLD = float(os.getenv("CLIP_SCORE_THRESHOLD", "0.35"))

# CUDA 设备配置 (用于 FFmpeg NVDEC 解码 + CUDA MOG2)
CUDA_DEVICE = os.getenv("CUDA_DEVICE", os.getenv("FFMPEG_NVDEC_DEVICE", "0"))

# MOG2 运动检测配置
USE_MOG2_FILTER = os.getenv("USE_MOG2_FILTER", "true").lower() == "true"
MOG2_FG_RATIO_THRESHOLD = float(os.getenv("MOG2_FG_RATIO_THRESHOLD", "0.015"))
MOG2_LOWRES_WIDTH = int(os.getenv("MOG2_LOWRES_WIDTH", "640"))
MOG2_LOWRES_HEIGHT = int(os.getenv("MOG2_LOWRES_HEIGHT", "360"))
MOG2_LOWRES_FPS = float(os.getenv("MOG2_LOWRES_FPS", "12"))
MOG2_DEBOUNCE_FRAMES = int(os.getenv("MOG2_DEBOUNCE_FRAMES", "3"))
MOG2_ALWAYS_SAMPLE_INTERVAL = float(os.getenv("MOG2_ALWAYS_SAMPLE_INTERVAL", "5.0"))

# 图片压缩配置
COMPRESS_MAX_SIZE = int(os.getenv("COMPRESS_MAX_SIZE", "640"))
COMPRESS_QUALITY = int(os.getenv("COMPRESS_QUALITY", "85"))

# CUDA MOG2 可用性检测 (启动时一次)
_CUDA_MOG2_AVAILABLE = None


def _check_cuda_mog2():
    """检测 cv2.cuda MOG2 是否可用 (失败时允许重试)"""
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
            # 验证 apply() 调用 (OpenCV 4.10 CUDA 需要 stream 参数)
            test_frame = cv2.cuda_GpuMat(np.zeros((64, 64), dtype=np.uint8))
            test_stream = cv2.cuda.Stream()
            test_mog2.apply(test_frame, -1, test_stream)
            _CUDA_MOG2_AVAILABLE = True
        else:
            _CUDA_MOG2_AVAILABLE = False
    except Exception as e:
        import logging
        logging.getLogger("semantic").warning(f"CUDA MOG2 检测失败 (将在下次任务重试): {e}")
        _CUDA_MOG2_AVAILABLE = None  # 不缓存失败，允许重试
        return False
    return _CUDA_MOG2_AVAILABLE


@dataclass
class FrameInfo:
    """帧信息"""
    frame_idx: int
    timestamp_sec: float
    frame_path: str
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    template_hit: bool = False
    tmpdir: str = None


@dataclass
class ClipInfo:
    """片段信息"""
    clip_id: int
    start_sec: float
    end_sec: float
    frames: List[FrameInfo]
    clip_score: float = 0.0
    keyframes: List[FrameInfo] = None

    def __post_init__(self):
        if self.keyframes is None:
            self.keyframes = []


@dataclass
class AnalysisResult:
    """分析结果"""
    judgment: str  # YES/NO/UNCERTAIN
    confidence: float
    reason: str
    raw_response: str = ""
    violation_type: str = None
    behavior_type: str = None
    marker_light_state: str = None
    extra: Dict[str, Any] = field(default_factory=dict)


class SemanticAnalyzerBase(ABC):
    """语义分析器基类"""

    def __init__(
        self,
        analysis_type: str,
        consumer_group: str,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        embedding_url: str = EMBEDDING_SERVICE_URL,
        vlm_proxy_url: str = VLM_PROXY_URL,
        results_dir: str = RESULTS_DIR
    ):
        self.analysis_type = analysis_type
        self.consumer_group = consumer_group
        self.consumer_name = os.getenv("HOSTNAME", f"{analysis_type}_{os.getpid()}")

        self.redis = RedisStreamClient(redis_host, redis_port)
        self.embedding_url = embedding_url.rstrip("/")
        self.vlm_proxy_url = vlm_proxy_url.rstrip("/")
        self.results_dir = Path(results_dir) / analysis_type
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(f"semantic-{analysis_type}")
        self.log = LogContext(self.logger, stage=f"semantic-{analysis_type}")

        self.task_count = 0
        self.running = True

        self.template_embeddings: Optional[np.ndarray] = None
        self.templates: List[str] = []

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.log.info(f"收到信号 {signum}，准备优雅退出")
        self.running = False

    @abstractmethod
    def get_templates(self) -> List[str]:
        """返回该分析类型的语义模板列表"""
        pass

    @abstractmethod
    def get_vlm_prompt(self) -> str:
        """返回该分析类型的 VLM prompt"""
        pass

    @abstractmethod
    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析 VLM 响应"""
        pass

    def should_process_task(self, task: VideoTask) -> bool:
        """每种分析器独立 consumer group，处理所有任务"""
        return True

    def compress_image(self, image_path: str, max_size: int = COMPRESS_MAX_SIZE, quality: int = COMPRESS_QUALITY) -> bytes:
        """压缩图片 (Embedding + VLM 共用)"""
        img = cv2.imread(image_path)
        if img is None:
            with open(image_path, "rb") as f:
                return f.read()
        h, w = img.shape[:2]
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buffer.tobytes()

    def load_template_embeddings(self):
        """加载语义模板嵌入"""
        self.templates = self.get_templates()
        if not self.templates:
            self.log.warning("无语义模板")
            return

        self.log.info(f"加载 {len(self.templates)} 个语义模板...")
        try:
            resp = requests.post(
                f"{self.embedding_url}/encode/text",
                json={"texts": self.templates},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            self.template_embeddings = np.array(data["embeddings"])
            self.log.info(f"模板嵌入加载完成: {self.template_embeddings.shape}")
        except Exception as e:
            self.log.error(f"加载模板嵌入失败: {e}")
            self.template_embeddings = None

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

    def extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """从视频提取帧 (FFmpeg NVDEC GPU 解码)"""
        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            self.log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames

        tmpdir = Path(tempfile.mkdtemp())
        output_pattern = str(tmpdir / "frame_%04d.jpg")

        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-hwaccel_device", CUDA_DEVICE,
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            output_pattern
        ]

        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
            )
            if result.returncode != 0:
                self.log.error(f"ffmpeg 抽帧失败: {result.stderr.decode()[-500:]}", job_id=job_id)
                return frames
        except Exception as e:
            self.log.error(f"抽帧异常: {e}", job_id=job_id)
            return frames

        frame_files = sorted(tmpdir.glob("frame_*.jpg"))
        for idx, frame_file in enumerate(frame_files):
            frames.append(FrameInfo(
                frame_idx=idx,
                timestamp_sec=idx / fps,
                frame_path=str(frame_file),
                tmpdir=str(tmpdir)
            ))

        self.log.info(f"抽取 {len(frames)} 帧 @ {fps} fps (GPU:{CUDA_DEVICE})", job_id=job_id, trace_id=trace_id)
        return frames

    def extract_frames_with_mog2(
        self,
        video_path: str,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """FFmpeg NVDEC 解码 + CUDA/CPU MOG2 运动检测

        流程:
        1. FFmpeg NVDEC 低分辨率管道 (640x360@12fps) -> rawvideo pipe
        2. CUDA MOG2 (或 CPU 回退) 运动检测 -> 记录触发时间点
        3. FFmpeg NVDEC select 表达式 -> 仅提取触发帧 (原始分辨率 JPEG)
        """
        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            self.log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames

        tmpdir = Path(tempfile.mkdtemp())
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

            # ===== 阶段1: FFmpeg NVDEC 低分辨率管道 + MOG2 =====
            t0 = time.time()

            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_device", CUDA_DEVICE,
                "-i", str(video_path),
                "-vf", f"scale={MOG2_LOWRES_WIDTH}:{MOG2_LOWRES_HEIGHT},fps={MOG2_LOWRES_FPS}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-v", "quiet", "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            frame_size = MOG2_LOWRES_WIDTH * MOG2_LOWRES_HEIGHT * 3

            if use_cuda_mog2:
                cv2.cuda.setDevice(int(CUDA_DEVICE))
                bg_subtractor = cv2.cuda.createBackgroundSubtractorMOG2(
                    history=400, varThreshold=16, detectShadows=True
                )
                cuda_stream = cv2.cuda.Stream()
                gpu_frame = cv2.cuda_GpuMat()
            else:
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=400, varThreshold=16, detectShadows=True
                )

            trigger_frame_indices = []
            frame_idx = 0
            motion_streak = 0
            last_force_ts = -1e9

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    MOG2_LOWRES_HEIGHT, MOG2_LOWRES_WIDTH, 3
                )
                timestamp = frame_idx / MOG2_LOWRES_FPS

                if use_cuda_mog2:
                    gpu_frame.upload(frame)
                    gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                    gpu_fg = bg_subtractor.apply(gpu_gray, -1, cuda_stream)
                    fg_mask = gpu_fg.download()
                else:
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

            proc.wait()
            t1 = time.time()

            self.log.info(
                f"阶段1 MOG2: {frame_idx} 帧 -> {len(trigger_frame_indices)} 触发点, "
                f"耗时={t1-t0:.1f}s",
                job_id=job_id, trace_id=trace_id
            )

            if not trigger_frame_indices:
                self.log.warning("MOG2 无触发点，跳过", job_id=job_id)
                return frames

            # ===== 阶段2: FFmpeg NVDEC 仅提取触发帧 =====
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
                        frame_path=frame_path,
                        tmpdir=str(tmpdir)
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
            # 清理可能残留的 ffmpeg 子进程
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass

        return frames

    def compute_embeddings(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """计算帧嵌入 (压缩后发送)"""
        if not frames:
            return frames

        images_b64 = []
        valid_indices = []

        for i, frame in enumerate(frames):
            try:
                img_bytes = self.compress_image(frame.frame_path)
                images_b64.append(base64.b64encode(img_bytes).decode())
                valid_indices.append(i)
            except Exception as e:
                self.log.warning(f"读取帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return frames

        try:
            resp = requests.post(
                f"{self.embedding_url}/encode/images",
                json={
                    "images": images_b64,
                    "job_id": job_id,
                    "trace_id": trace_id
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = np.array(data["embeddings"])

            for i, idx in enumerate(valid_indices):
                frames[idx].embedding = embeddings[i]

            self.log.info(f"计算 {len(embeddings)} 帧嵌入", job_id=job_id, trace_id=trace_id)

        except Exception as e:
            self.log.error(f"计算嵌入失败: {e}", job_id=job_id, trace_id=trace_id)

        return frames

    def compute_similarity_scores(
        self,
        frames: List[FrameInfo],
        threshold: float = 0.3,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """计算帧与模板的相似度"""
        if self.template_embeddings is None or not frames:
            return frames

        for frame in frames:
            if frame.embedding is None:
                continue

            frame_norm = frame.embedding / np.linalg.norm(frame.embedding)
            template_norms = self.template_embeddings / np.linalg.norm(
                self.template_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(template_norms, frame_norm)

            max_sim = float(np.max(similarities))
            frame.similarity_score = max_sim
            frame.template_hit = max_sim > threshold

        return frames

    def cluster_frames_to_clips(
        self,
        frames: List[FrameInfo],
        threshold: float = 0.08,
        min_gap_sec: float = 5.0,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[ClipInfo]:
        """基于相似度分数聚类帧为片段"""
        if not frames:
            return []

        high_score_frames = [f for f in frames if f.similarity_score >= threshold]

        if not high_score_frames:
            self.log.info("无高分帧，跳过聚类", job_id=job_id, trace_id=trace_id)
            return []

        clips = []
        current_clip_frames = [high_score_frames[0]]

        for frame in high_score_frames[1:]:
            last_frame = current_clip_frames[-1]
            gap = frame.timestamp_sec - last_frame.timestamp_sec

            if gap <= min_gap_sec:
                current_clip_frames.append(frame)
            else:
                clips.append(ClipInfo(
                    clip_id=len(clips),
                    start_sec=current_clip_frames[0].timestamp_sec,
                    end_sec=current_clip_frames[-1].timestamp_sec,
                    frames=current_clip_frames,
                    clip_score=max(f.similarity_score for f in current_clip_frames)
                ))
                current_clip_frames = [frame]

        if current_clip_frames:
            clips.append(ClipInfo(
                clip_id=len(clips),
                start_sec=current_clip_frames[0].timestamp_sec,
                end_sec=current_clip_frames[-1].timestamp_sec,
                frames=current_clip_frames,
                clip_score=max(f.similarity_score for f in current_clip_frames)
            ))

        self.log.info(f"聚类生成 {len(clips)} 个片段", job_id=job_id, trace_id=trace_id)
        return clips

    def select_keyframes(
        self,
        clip: ClipInfo,
        max_frames: int = 12,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """选择关键帧（按相似度排序）"""
        sorted_frames = sorted(
            clip.frames,
            key=lambda f: f.similarity_score,
            reverse=True
        )
        return sorted_frames[:max_frames]

    def call_vlm(
        self,
        keyframes: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> AnalysisResult:
        """调用 VLM 进行分析 (压缩后发送)"""
        if not keyframes:
            return AnalysisResult(judgment="NO", confidence=0.0, reason="无关键帧")

        images_b64 = []
        for frame in keyframes:
            try:
                img_bytes = self.compress_image(frame.frame_path)
                images_b64.append(base64.b64encode(img_bytes).decode())
            except Exception as e:
                self.log.warning(f"读取关键帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return AnalysisResult(judgment="NO", confidence=0.0, reason="无法读取关键帧")

        prompt = self.get_vlm_prompt()

        content = [{"type": "text", "text": prompt}]
        for b64_img in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

        try:
            resp = requests.post(
                f"{self.vlm_proxy_url}/chat/completions",
                json={
                    "model": "qwen3-vl-32b",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 800,
                    "temperature": 0.1
                },
                headers={
                    "X-Trace-Id": trace_id,
                    "X-Job-Id": job_id
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            result = self.parse_vlm_response(response_text)
            result.raw_response = response_text

            self.log.info(
                f"VLM 判断: {result.judgment} (conf={result.confidence})",
                job_id=job_id, trace_id=trace_id
            )
            return result

        except Exception as e:
            self.log.error(f"VLM 调用失败: {e}", job_id=job_id, trace_id=trace_id)
            return AnalysisResult(judgment="ERROR", confidence=0.0, reason=str(e))

    def save_result(
        self,
        task: VideoTask,
        clips: List[ClipInfo],
        analysis_result: AnalysisResult,
        processing_time_sec: float
    ) -> Path:
        """保存结果到文件"""
        result = {
            "job_id": task.job_id,
            "camera_id": task.camera_id,
            "trace_id": task.trace_id,
            "analysis_type": self.analysis_type,
            "window_path": task.window_path,
            "seg_end_ts": task.seg_end_ts,
            "created_at": task.created_at,
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "processing_time_sec": processing_time_sec,
            "clips": [
                {
                    "clip_id": c.clip_id,
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                    "clip_score": c.clip_score,
                    "frame_count": len(c.frames)
                }
                for c in clips
            ],
            "analysis_result": {
                "judgment": analysis_result.judgment,
                "confidence": analysis_result.confidence,
                "reason": analysis_result.reason,
                "violation_type": analysis_result.violation_type,
                "behavior_type": analysis_result.behavior_type,
                "marker_light_state": analysis_result.marker_light_state,
                "extra": analysis_result.extra
            },
            "is_positive": analysis_result.judgment == "YES"
        }

        output_dir = self.results_dir / task.camera_id
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task.job_id}.result.json.gz"
        tmp_path = output_dir / f"{filename}.tmp"
        final_path = output_dir / filename

        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.rename(tmp_path, final_path)

            self.log.info(f"结果已保存: {final_path}", job_id=task.job_id, trace_id=task.trace_id)
            return final_path

        except Exception as e:
            self.log.error(f"保存结果失败: {e}", job_id=task.job_id)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def process_task(self, task: VideoTask) -> bool:
        """处理单个任务"""
        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id

        self.log.info(
            f"开始处理任务 [{self.analysis_type}]: {task.window_path}",
            job_id=job_id, trace_id=trace_id
        )

        try:
            self.redis.set_task_status(job_id, "processing", self.consumer_name)

            # 1. 抽帧
            if USE_MOG2_FILTER:
                frames = self.extract_frames_with_mog2(task.window_path, job_id=job_id, trace_id=trace_id)
            else:
                frames = self.extract_frames(task.window_path, fps=1.0, job_id=job_id, trace_id=trace_id)

            if not frames:
                self.log.warning("抽帧为空，标记完成", job_id=job_id)
                self.redis.set_task_status(job_id, "done", self.consumer_name)
                return True

            # 2. 计算嵌入
            frames = self.compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

            # 3. 计算相似度
            frames = self.compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

            # 4. 聚类
            clips = self.cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

            # 5. VLM 分析
            analysis_result = AnalysisResult(judgment="NO", confidence=1.0, reason="无可疑片段")

            if clips:
                best_clip = max(clips, key=lambda c: c.clip_score)

                if best_clip.clip_score >= CLIP_SCORE_THRESHOLD:
                    self.log.info(
                        f"最高分片段 {best_clip.clip_score:.3f} >= 阈值 {CLIP_SCORE_THRESHOLD}, 调用 VLM",
                        job_id=job_id, trace_id=trace_id
                    )
                    keyframes = self.select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                    best_clip.keyframes = keyframes
                    analysis_result = self.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)
                else:
                    self.log.info(
                        f"最高分片段 {best_clip.clip_score:.3f} < 阈值 {CLIP_SCORE_THRESHOLD}, 跳过 VLM",
                        job_id=job_id, trace_id=trace_id
                    )
                    analysis_result = AnalysisResult(
                        judgment="NO", confidence=1.0,
                        reason=f"最高分 {best_clip.clip_score:.3f} 低于阈值"
                    )

            # 6. 保存结果
            processing_time = time.time() - start_time
            result_path = self.save_result(task, clips, analysis_result, processing_time)

            # 7. 发送结果任务
            result_task = ResultTask(
                event_time=task.seg_end_ts or time.time(),
                job_id=job_id,
                camera_id=task.camera_id,
                trace_id=trace_id,
                result_path=str(result_path),
                is_accident=analysis_result.judgment == "YES" and self.analysis_type == "accident",
                is_positive=analysis_result.judgment == "YES",
                confidence=analysis_result.confidence,
                seg_end_ts=task.seg_end_ts,
                analysis_type=self.analysis_type,
                violation_type=analysis_result.violation_type,
                behavior_type=analysis_result.behavior_type
            )
            self.redis.add_result_task(result_task)

            # 8. 清理临时文件
            self._cleanup_frames(frames)

            # 9. 更新状态
            self.redis.set_task_status(job_id, "done", self.consumer_name)

            self.log.info(
                f"任务完成: {job_id}, 耗时={processing_time:.1f}s, 判断={analysis_result.judgment}",
                job_id=job_id, trace_id=trace_id
            )
            return True

        except Exception as e:
            self.log.error(f"任务处理失败: {e}", job_id=job_id, trace_id=trace_id)
            self.redis.set_task_status(job_id, "failed", self.consumer_name)
            return False

    def _cleanup_frames(self, frames: List[FrameInfo]):
        """清理临时帧文件"""
        import shutil
        cleaned_dirs = set()
        for frame in frames:
            if frame.tmpdir and frame.tmpdir not in cleaned_dirs:
                try:
                    shutil.rmtree(frame.tmpdir)
                    cleaned_dirs.add(frame.tmpdir)
                except Exception:
                    pass

    def run(self):
        """主循环"""
        self.log.info(f"启动 Semantic Analyzer [{self.analysis_type}]: {self.consumer_name}")

        if _check_cuda_mog2():
            self.log.info("CUDA MOG2 可用")
        else:
            self.log.warning("CUDA MOG2 不可用，回退到 CPU MOG2")

        self.load_template_embeddings()

        while self.running:
            if MAX_TASKS_BEFORE_EXIT > 0 and self.task_count >= MAX_TASKS_BEFORE_EXIT:
                self.log.info(f"达到自愈阈值 ({MAX_TASKS_BEFORE_EXIT})，准备退出")
                break

            try:
                tasks = self.redis.read_video_tasks(
                    group=self.consumer_group,
                    consumer=self.consumer_name,
                    count=1,
                    block=5000
                )

                for msg_id, task in tasks:
                    if not self.running:
                        break

                    if not self.should_process_task(task):
                        self.redis.ack_video_task(self.consumer_group, msg_id)
                        continue

                    if self.process_task(task):
                        self.redis.ack_video_task(self.consumer_group, msg_id)
                        self.task_count += 1
                        self.log.info(
                            f"已处理 {self.task_count}/{MAX_TASKS_BEFORE_EXIT} 个任务",
                            job_id=task.job_id
                        )

            except Exception as e:
                self.log.error(f"主循环异常: {e}")
                time.sleep(5)

        self.log.info(f"Semantic Analyzer [{self.analysis_type}] 退出")
