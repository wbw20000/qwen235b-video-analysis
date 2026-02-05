#!/usr/bin/env python3
from urllib.parse import quote
"""
Semantic Analyzer - 唯一编排者（P0 必做）
- 消费 video_tasks 队列
- HTTP 调用 Embedding Service
- 执行聚类 + 关键帧选择
- HTTP 调用 VLM Proxy
- 写入 result_tasks 队列

架构角色: 唯一的 orchestrator，简化消息流
"""
import os
import sys
import time
import signal
import json
import base64
import hashlib
import cv2

def compress_image_for_vlm(image_path: str, max_size: int = 640, quality: int = 85) -> bytes:
    """压缩图片以发送给VLM (使用cv2)"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    return buffer.tobytes()

import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
import numpy as np

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, VideoTask, ResultTask

# 配置
REDIS_HOST = os.getenv("REDIS_HOST", "10.104.10.203")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://10.96.40.236:8080")
VLM_PROXY_URL = os.getenv("VLM_PROXY_URL", "http://10.99.201.98:8001")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")
CONSUMER_GROUP = "semantic_analyzers"
CONSUMER_NAME = os.getenv("HOSTNAME", f"analyzer_{os.getpid()}")
MAX_TASKS_BEFORE_EXIT = int(os.getenv("MAX_TASKS", "50"))
TASK_TIMEOUT_SEC = 300  # 单任务超时
CLIP_SCORE_THRESHOLD = float(os.getenv("CLIP_SCORE_THRESHOLD", "0.35"))  # VLM调用阈值
FFMPEG_NVDEC_ENABLED = os.getenv("FFMPEG_NVDEC_ENABLED", "true").lower() == "true"  # GPU解码开关
FFMPEG_NVDEC_DEVICE = os.getenv("FFMPEG_NVDEC_DEVICE", "0")  # GPU设备ID

# 事故模板（用于 SigLIP 相似度匹配）- 中文模板与本地 config.py 保持一致
ACCIDENT_TEMPLATES = [
    # 机动车之间事故
    "路口画面两辆机动车发生碰撞",
    "监控视频中汽车之间发生碰撞",
    "机动车追尾前车",
    "两车相撞，车辆受损",
    "十字路口车辆侧面碰撞",
    # 机动车与二轮车事故
    "路口画面机动车与二轮车发生碰撞",
    "监控视频中汽车与电动车发生碰撞",
    "机动车与自行车发生接触",
    "汽车与摩托车碰撞后骑车人摔倒",
    "路口机动车撞到电动车",
    # 机动车与行人事故
    "路口画面机动车与行人发生碰撞",
    "监控视频中汽车撞到行人",
    "机动车在人行横道撞到行人",
    "车辆与行人发生交通事故",
    # 多车事故
    "路口画面多车连环相撞",
    "监控视频中三辆或以上车辆连环相撞",
    "多车连撞事故",
    "车辆连环追尾事故",
    # 肇事逃逸
    "发生交通事故后车辆逃离现场",
    "肇事车辆逃逸",
    "碰撞后未停车离开",
    "事故后司机驾车逃逸",
]

logger = setup_logger("semantic-analyzer")
log = LogContext(logger, stage="semantic-analyzer")


@dataclass
class FrameInfo:
    """帧信息"""
    frame_idx: int
    timestamp_sec: float
    frame_path: str
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    accident_template_hit: bool = False
    tmpdir: str = None  # 临时目录路径，用于清理


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


class SemanticAnalyzer:
    """语义分析器 - 唯一编排者"""

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        embedding_url: str = EMBEDDING_SERVICE_URL,
        vlm_proxy_url: str = VLM_PROXY_URL,
        results_dir: str = RESULTS_DIR
    ):
        self.redis = RedisStreamClient(redis_host, redis_port)
        self.embedding_url = embedding_url.rstrip("/")
        self.vlm_proxy_url = vlm_proxy_url.rstrip("/")
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.task_count = 0
        self.running = True

        # 事故模板嵌入（启动时计算）
        self.template_embeddings: Optional[np.ndarray] = None

        # 信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info(f"收到信号 {signum}，准备优雅退出")
        self.running = False

    def _load_template_embeddings(self):
        """加载事故模板嵌入"""
        log.info("加载事故模板嵌入...")
        try:
            resp = requests.post(
                f"{self.embedding_url}/encode/text",
                json={"texts": ACCIDENT_TEMPLATES},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            self.template_embeddings = np.array(data["embeddings"])
            log.info(f"模板嵌入加载完成: {self.template_embeddings.shape}")
        except Exception as e:
            log.error(f"加载模板嵌入失败: {e}")
            self.template_embeddings = None

    def _extract_frames(
        self,
        video_path: str,
        fps: float = 1.0,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """
        从视频提取帧
        使用 ffmpeg 抽帧到持久临时目录
        """
        import tempfile
        import subprocess

        frames = []
        video_path = Path(video_path)

        if not video_path.exists():
            log.error(f"视频不存在: {video_path}", job_id=job_id)
            return frames

        # 创建持久临时目录 (帧处理完成后由调用方清理)
        tmpdir = Path(tempfile.mkdtemp())

        # 使用 ffmpeg 抽帧 (支持 NVDEC GPU 解码加速)
        output_pattern = str(tmpdir / "frame_%04d.jpg")

        if FFMPEG_NVDEC_ENABLED:
            # GPU 解码: -hwaccel cuda 启用 NVIDIA 硬件加速
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
            # CPU 解码 (回退模式)
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
                log.error(f"ffmpeg 抽帧失败: {result.stderr.decode()[-500:]}", job_id=job_id)
                return frames
        except Exception as e:
            log.error(f"抽帧异常: {e}", job_id=job_id)
            return frames

        # 收集帧文件
        frame_files = sorted(tmpdir.glob("frame_*.jpg"))
        for idx, frame_file in enumerate(frame_files):
            frames.append(FrameInfo(
                frame_idx=idx,
                timestamp_sec=idx / fps,
                frame_path=str(frame_file),
                tmpdir=str(tmpdir)  # 存储临时目录以便后续清理
            ))

        decode_mode = f"GPU:{FFMPEG_NVDEC_DEVICE}" if FFMPEG_NVDEC_ENABLED else "CPU"
        log.info(f"抽取 {len(frames)} 帧 @ {fps} fps ({decode_mode})", job_id=job_id, trace_id=trace_id)

        return frames

    def _compute_embeddings(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """
        计算帧嵌入
        调用 Embedding Service
        """
        if not frames:
            return frames

        # 读取帧图像并编码为 base64
        images_b64 = []
        valid_indices = []

        for i, frame in enumerate(frames):
            try:
                # 使用压缩函数
                img_bytes = compress_image_for_vlm(frame.frame_path)
                images_b64.append(base64.b64encode(img_bytes).decode())
                valid_indices.append(i)
            except Exception as e:
                log.warning(f"读取帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return frames

        # 调用 Embedding Service
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

            # 更新帧嵌入
            for i, idx in enumerate(valid_indices):
                frames[idx].embedding = embeddings[i]

            log.info(f"计算 {len(embeddings)} 帧嵌入", job_id=job_id, trace_id=trace_id)

        except Exception as e:
            log.error(f"计算嵌入失败: {e}", job_id=job_id, trace_id=trace_id)

        return frames

    def _compute_similarity_scores(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """
        计算帧与事故模板的相似度
        """
        if self.template_embeddings is None or not frames:
            return frames

        for frame in frames:
            if frame.embedding is None:
                continue

            # 计算与所有模板的相似度
            frame_norm = frame.embedding / np.linalg.norm(frame.embedding)
            template_norms = self.template_embeddings / np.linalg.norm(
                self.template_embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(template_norms, frame_norm)

            # 取最大相似度
            max_sim = float(np.max(similarities))
            frame.similarity_score = max_sim
            log.info(f"帧 {frame.frame_idx} 相似度: {max_sim:.4f}", job_id=job_id, trace_id=trace_id)
            frame.accident_template_hit = max_sim > 0.3  # 阈值

        return frames

    def _cluster_frames_to_clips(
        self,
        frames: List[FrameInfo],
        threshold: float = 0.08,
        min_gap_sec: float = 5.0,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[ClipInfo]:
        """
        基于相似度分数聚类帧为片段
        """
        if not frames:
            return []

        # 筛选高分帧
        high_score_frames = [f for f in frames if f.similarity_score >= threshold]

        if not high_score_frames:
            log.info("无高分帧，跳过聚类", job_id=job_id, trace_id=trace_id)
            return []

        # 按时间聚类
        clips = []
        current_clip_frames = [high_score_frames[0]]

        for frame in high_score_frames[1:]:
            last_frame = current_clip_frames[-1]
            gap = frame.timestamp_sec - last_frame.timestamp_sec

            if gap <= min_gap_sec:
                current_clip_frames.append(frame)
            else:
                # 保存当前片段
                clips.append(ClipInfo(
                    clip_id=len(clips),
                    start_sec=current_clip_frames[0].timestamp_sec,
                    end_sec=current_clip_frames[-1].timestamp_sec,
                    frames=current_clip_frames,
                    clip_score=max(f.similarity_score for f in current_clip_frames)
                ))
                current_clip_frames = [frame]

        # 保存最后一个片段
        if current_clip_frames:
            clips.append(ClipInfo(
                clip_id=len(clips),
                start_sec=current_clip_frames[0].timestamp_sec,
                end_sec=current_clip_frames[-1].timestamp_sec,
                frames=current_clip_frames,
                clip_score=max(f.similarity_score for f in current_clip_frames)
            ))

        log.info(f"聚类生成 {len(clips)} 个片段", job_id=job_id, trace_id=trace_id)
        return clips

    def _select_keyframes(
        self,
        clip: ClipInfo,
        max_frames: int = 12,
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """
        选择关键帧（按相似度排序，取前 N 帧）
        """
        sorted_frames = sorted(
            clip.frames,
            key=lambda f: f.similarity_score,
            reverse=True
        )
        return sorted_frames[:max_frames]

    def _call_vlm(
        self,
        keyframes: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> Dict[str, Any]:
        """
        调用 VLM 进行事故分析
        """
        if not keyframes:
            return {"judgment": "NO", "confidence": 0.0, "reason": "无关键帧"}

        # 准备图像数据
        images_b64 = []
        for frame in keyframes:
            try:
                # 使用压缩函数
                img_bytes = compress_image_for_vlm(frame.frame_path)
                images_b64.append(base64.b64encode(img_bytes).decode())
            except Exception as e:
                log.warning(f"读取关键帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return {"judgment": "NO", "confidence": 0.0, "reason": "无法读取关键帧"}

        # 构建 VLM 请求
        prompt = """分析这些视频帧，判断是否发生了交通事故。

请按以下格式回答：
1. 判断: YES（确定发生事故）/ NO（未发生事故）/ UNCERTAIN（不确定）
2. 置信度: 0.0-1.0
3. 原因: 简要说明判断依据

注意事项：
- 观察车辆位置、姿态、运动轨迹
- 注意碰撞痕迹、车辆变形、人员倒地等迹象
- 区分正常行驶和事故场景"""

        # 构建消息
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
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                headers={
                    "X-Trace-Id": trace_id,
                    "X-Job-Id": quote(job_id, safe="")
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            # 解析响应
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 简单解析
            judgment = "UNCERTAIN"
            confidence = 0.5
            reason = response_text

            if "YES" in response_text.upper()[:50]:
                judgment = "YES"
                confidence = 0.8
            elif "NO" in response_text.upper()[:50]:
                judgment = "NO"
                confidence = 0.7

            log.info(
                f"VLM 判断: {judgment} (conf={confidence})",
                job_id=job_id,
                trace_id=trace_id
            )

            return {
                "judgment": judgment,
                "confidence": confidence,
                "reason": reason,
                "raw_response": response_text
            }

        except Exception as e:
            log.error(f"VLM 调用失败: {e}", job_id=job_id, trace_id=trace_id)
            return {
                "judgment": "ERROR",
                "confidence": 0.0,
                "reason": str(e)
            }

    def _save_result(
        self,
        task: VideoTask,
        clips: List[ClipInfo],
        vlm_result: Dict[str, Any],
        processing_time_sec: float
    ) -> Path:
        """
        保存结果到文件（原子写入 + gzip 压缩）
        """
        result = {
            "job_id": task.job_id,
            "camera_id": task.camera_id,
            "trace_id": task.trace_id,
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
            "vlm_result": vlm_result,
            "is_accident": vlm_result.get("judgment") == "YES"
        }

        # 输出路径
        output_dir = self.results_dir / task.camera_id
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{task.job_id}.result.json.gz"
        tmp_path = output_dir / f"{filename}.tmp"
        final_path = output_dir / filename

        # 原子写入
        try:
            with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # fsync + rename
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.rename(tmp_path, final_path)

            log.info(f"结果已保存: {final_path}", job_id=task.job_id, trace_id=task.trace_id)
            return final_path

        except Exception as e:
            log.error(f"保存结果失败: {e}", job_id=task.job_id)
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def process_task(self, task: VideoTask) -> bool:
        """处理单个任务"""
        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id

        log.info(
            f"开始处理任务: {task.window_path}",
            job_id=job_id,
            trace_id=trace_id
        )

        try:
            # 更新状态
            self.redis.set_task_status(job_id, "processing", CONSUMER_NAME)

            # 1. 抽帧
            frames = self._extract_frames(task.window_path, fps=1.0, job_id=job_id, trace_id=trace_id)
            if not frames:
                log.warning("抽帧为空，标记完成", job_id=job_id)
                self.redis.set_task_status(job_id, "done", CONSUMER_NAME)
                return True

            # 2. 计算嵌入
            frames = self._compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

            # 3. 计算相似度
            frames = self._compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

            # 4. 聚类
            clips = self._cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

            # 5. 选择关键帧 + VLM 分析
            vlm_result = {"judgment": "NO", "confidence": 1.0, "reason": "无可疑片段"}

            if clips:
                # 选择最高分片段
                best_clip = max(clips, key=lambda c: c.clip_score)

                # 只有超过阈值才调用 VLM
                if best_clip.clip_score >= CLIP_SCORE_THRESHOLD:
                    log.info(f"最高分片段 {best_clip.clip_score:.3f} >= 阈值 {CLIP_SCORE_THRESHOLD}, 调用 VLM",
                             job_id=job_id, trace_id=trace_id)
                    keyframes = self._select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                    best_clip.keyframes = keyframes
                    vlm_result = self._call_vlm(keyframes, job_id=job_id, trace_id=trace_id)
                else:
                    log.info(f"最高分片段 {best_clip.clip_score:.3f} < 阈值 {CLIP_SCORE_THRESHOLD}, 跳过 VLM",
                             job_id=job_id, trace_id=trace_id)
                    vlm_result = {"judgment": "NO", "confidence": 1.0, "reason": f"最高分 {best_clip.clip_score:.3f} 低于阈值"}

            # 6. 保存结果
            processing_time = time.time() - start_time
            result_path = self._save_result(task, clips, vlm_result, processing_time)

            # 7. 发送结果任务
            result_task = ResultTask(
                event_time=task.seg_end_ts or time.time(),
                job_id=job_id,
                camera_id=task.camera_id,
                trace_id=trace_id,
                result_path=str(result_path),
                is_accident=vlm_result.get("judgment") == "YES",
                confidence=vlm_result.get("confidence", 0.0),
                seg_end_ts=task.seg_end_ts,
                processing_time_sec=processing_time
            )
            self.redis.add_result_task(result_task)

            # 8. 更新状态
            self.redis.set_task_status(job_id, "done", CONSUMER_NAME)

            log.info(
                f"任务完成: {job_id}, 耗时={processing_time:.1f}s, 判断={vlm_result.get('judgment')}",
                job_id=job_id,
                trace_id=trace_id
            )

            return True

        except Exception as e:
            log.error(f"任务处理失败: {e}", job_id=job_id, trace_id=trace_id)
            self.redis.set_task_status(job_id, "failed", CONSUMER_NAME)
            return False

    def run(self):
        """主循环"""
        log.info(f"启动 Semantic Analyzer: {CONSUMER_NAME}")

        # 加载模板嵌入
        self._load_template_embeddings()

        while self.running:
            # 检查自愈阈值
            if MAX_TASKS_BEFORE_EXIT > 0 and self.task_count >= MAX_TASKS_BEFORE_EXIT:
                log.info(f"达到自愈阈值 ({MAX_TASKS_BEFORE_EXIT})，准备退出")
                break

            try:
                # 读取任务
                tasks = self.redis.read_video_tasks(
                    consumer_group=CONSUMER_GROUP,
                    consumer_name=CONSUMER_NAME,
                    count=1,
                    block_ms=5000
                )

                for msg_id, task in tasks:
                    if not self.running:
                        break

                    if self.process_task(task):
                        self.redis.ack_video_task(CONSUMER_GROUP, msg_id)
                        self.task_count += 1
                        log.info(
                            f"已处理 {self.task_count}/{MAX_TASKS_BEFORE_EXIT} 个任务",
                            job_id=task.job_id
                        )

            except Exception as e:
                log.error(f"主循环异常: {e}")
                time.sleep(5)

        log.info("Semantic Analyzer 退出")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Analyzer")
    parser.add_argument("--redis-host", default=REDIS_HOST, help="Redis 主机")
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT, help="Redis 端口")
    parser.add_argument("--embedding-url", default=EMBEDDING_SERVICE_URL, help="Embedding Service URL")
    parser.add_argument("--vlm-proxy-url", default=VLM_PROXY_URL, help="VLM Proxy URL")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="结果输出目录")
    args = parser.parse_args()

    analyzer = SemanticAnalyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
