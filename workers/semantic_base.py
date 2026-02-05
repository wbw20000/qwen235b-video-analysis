#!/usr/bin/env python3
"""
Semantic Analyzer 基类 - 多语义检测共享逻辑
支持: accident, mv_violation, ebike_violation, ads_behavior

设计原则:
- 不修改现有 accident 检测逻辑
- 复用 Embedding + VLM 调用链
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
FFMPEG_NVDEC_ENABLED = os.getenv("FFMPEG_NVDEC_ENABLED", "true").lower() == "true"
FFMPEG_NVDEC_DEVICE = os.getenv("FFMPEG_NVDEC_DEVICE", "0")

# MOG2 运动检测配置
USE_MOG2_FILTER = os.getenv("USE_MOG2_FILTER", "true").lower() == "true"
MOG2_FG_RATIO_THRESHOLD = float(os.getenv("MOG2_FG_RATIO_THRESHOLD", "0.015"))
MOG2_LOWRES_WIDTH = int(os.getenv("MOG2_LOWRES_WIDTH", "640"))
MOG2_LOWRES_HEIGHT = int(os.getenv("MOG2_LOWRES_HEIGHT", "360"))
MOG2_LOWRES_FPS = float(os.getenv("MOG2_LOWRES_FPS", "12"))
MOG2_DEBOUNCE_FRAMES = int(os.getenv("MOG2_DEBOUNCE_FRAMES", "3"))
MOG2_ALWAYS_SAMPLE_INTERVAL = float(os.getenv("MOG2_ALWAYS_SAMPLE_INTERVAL", "5.0"))
# 移除 MOG2_MAX_FRAMES 限制 - GPU 已扩展，无需采样限制


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
    violation_type: str = None  # 违法类型
    behavior_type: str = None  # 行为类型
    marker_light_state: str = None  # 示廓灯状态 (ads_behavior)
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

        # 模板嵌入（子类加载）
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
        """判断是否处理该任务（基于 analysis_type）"""
        return task.analysis_type == self.analysis_type

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

    def extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """从视频提取帧（复用原逻辑，支持 NVDEC）"""
        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            self.log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames

        tmpdir = Path(tempfile.mkdtemp())
        output_pattern = str(tmpdir / "frame_%04d.jpg")

        if FFMPEG_NVDEC_ENABLED:
            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-hwaccel_device", FFMPEG_NVDEC_DEVICE,
                "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                output_pattern
            ]
        else:
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"fps={fps}",
                "-q:v", "2",
                output_pattern
            ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
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

        decode_mode = f"GPU:{FFMPEG_NVDEC_DEVICE}" if FFMPEG_NVDEC_ENABLED else "CPU"
        self.log.info(f"抽取 {len(frames)} 帧 @ {fps} fps ({decode_mode})", job_id=job_id, trace_id=trace_id)

        return frames

    def extract_frames_with_mog2(
        self,
        video_path: str,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """使用 MOG2 运动检测提取关键帧（取代 1fps 固定抽帧）

        流程：
        1. 低分辨率 (640×360 @ 12fps) 读取视频
        2. MOG2 检测运动触发点 (fg_ratio >= 0.015)
        3. 仅在触发点提取高清帧

        优势：过滤 90%+ 静止画面，只处理有运动的帧
        """
        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            self.log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames

        # 创建临时目录
        tmpdir = Path(tempfile.mkdtemp())

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.log.error(f"无法打开视频: {video_path}", job_id=job_id)
                return frames

            # 获取视频属性
            orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算低分辨率采样间隔
            sample_interval = max(1, int(orig_fps / MOG2_LOWRES_FPS))

            self.log.info(
                f"MOG2 运动检测: {orig_width}x{orig_height}@{orig_fps:.1f}fps -> "
                f"{MOG2_LOWRES_WIDTH}x{MOG2_LOWRES_HEIGHT}@{MOG2_LOWRES_FPS}fps, "
                f"interval={sample_interval}",
                job_id=job_id, trace_id=trace_id
            )

            # 初始化 MOG2
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=16, detectShadows=True
            )

            motion_streak = 0
            trigger_timestamps = []
            last_force_ts = -1e9
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 跳帧采样
                if frame_idx % sample_interval != 0:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / orig_fps

                # 缩小到低分辨率
                lowres = cv2.resize(frame, (MOG2_LOWRES_WIDTH, MOG2_LOWRES_HEIGHT))
                gray = cv2.cvtColor(lowres, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                # MOG2 运动检测
                fg_mask = bg_subtractor.apply(gray)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size

                # 检测运动触发
                if fg_ratio >= MOG2_FG_RATIO_THRESHOLD:
                    motion_streak += 1
                else:
                    motion_streak = 0

                triggered = False

                # 连续帧触发（防抖）
                if motion_streak >= MOG2_DEBOUNCE_FRAMES:
                    triggered = True
                    motion_streak = 0

                # 强制采样兜底
                if timestamp - last_force_ts >= MOG2_ALWAYS_SAMPLE_INTERVAL:
                    triggered = True
                    last_force_ts = timestamp

                if triggered:
                    trigger_timestamps.append((frame_idx, timestamp, fg_ratio))

                frame_idx += 1

            cap.release()

            self.log.info(
                f"MOG2 触发 {len(trigger_timestamps)} 个时间点 (共 {total_frames} 帧)",
                job_id=job_id, trace_id=trace_id
            )

            if not trigger_timestamps:
                self.log.warning("MOG2 无触发点，跳过", job_id=job_id)
                return frames

            # GPU 已扩展，不再限制帧数

            # 第二遍：仅在触发点提取高清帧
            cap = cv2.VideoCapture(str(video_path))
            trigger_set = {t[0] for t in trigger_timestamps}

            frame_idx = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx in trigger_set:
                    # 保存高清帧
                    frame_path = str(tmpdir / f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    timestamp = frame_idx / orig_fps
                    frames.append(FrameInfo(
                        frame_idx=saved_count,
                        timestamp_sec=timestamp,
                        frame_path=frame_path,
                        tmpdir=str(tmpdir)
                    ))
                    saved_count += 1

                frame_idx += 1

            cap.release()

            self.log.info(
                f"MOG2 提取 {len(frames)} 帧 (过滤 {100 * (1 - len(frames) / max(1, total_frames / sample_interval)):.1f}% 静止帧)",
                job_id=job_id, trace_id=trace_id
            )

        except Exception as e:
            self.log.error(f"MOG2 抽帧异常: {e}", job_id=job_id, trace_id=trace_id)
            import traceback
            self.log.error(traceback.format_exc())

        return frames

    def compute_embeddings(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """计算帧嵌入"""
        if not frames:
            return frames

        images_b64 = []
        valid_indices = []

        for i, frame in enumerate(frames):
            try:
                with open(frame.frame_path, "rb") as f:
                    img_bytes = f.read()
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
        """调用 VLM 进行分析"""
        if not keyframes:
            return AnalysisResult(
                judgment="NO",
                confidence=0.0,
                reason="无关键帧"
            )

        images_b64 = []
        for frame in keyframes:
            try:
                with open(frame.frame_path, "rb") as f:
                    img_bytes = f.read()
                images_b64.append(base64.b64encode(img_bytes).decode())
            except Exception as e:
                self.log.warning(f"读取关键帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return AnalysisResult(
                judgment="NO",
                confidence=0.0,
                reason="无法读取关键帧"
            )

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
                job_id=job_id,
                trace_id=trace_id
            )

            return result

        except Exception as e:
            self.log.error(f"VLM 调用失败: {e}", job_id=job_id, trace_id=trace_id)
            return AnalysisResult(
                judgment="ERROR",
                confidence=0.0,
                reason=str(e)
            )

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
        """处理单个任务（子类可覆盖）"""
        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id

        self.log.info(
            f"开始处理任务 [{self.analysis_type}]: {task.window_path}",
            job_id=job_id,
            trace_id=trace_id
        )

        try:
            self.redis.set_task_status(job_id, "processing", self.consumer_name)

            # 1. 抽帧（MOG2 运动检测 或 1fps 固定抽帧）
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
            analysis_result = AnalysisResult(
                judgment="NO",
                confidence=1.0,
                reason="无可疑片段"
            )

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
                        judgment="NO",
                        confidence=1.0,
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
                job_id=job_id,
                trace_id=trace_id
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

                    # 只处理匹配 analysis_type 的任务
                    if not self.should_process_task(task):
                        self.log.info(
                            f"跳过任务: analysis_type={task.analysis_type} != {self.analysis_type}",
                            job_id=task.job_id
                        )
                        # ACK 不匹配的消息（每个 consumer group 独立，无法让其他分析器处理）
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
