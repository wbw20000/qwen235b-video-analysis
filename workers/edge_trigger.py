#!/usr/bin/env python3
"""
Edge Trigger - 边缘端事故预检测触发器

功能:
- 接收滑动窗口视频 (60秒窗口, 5秒滑动)
- 使用轻量 SigLIP 计算帧与事故模板的相似度
- 超过阈值时触发事件上传到中心云
- 支持 CPU-only 模式 (ONNX Runtime)

部署位置: 边缘云 (靠近摄像头)
资源需求: 1 CPU, 2G Mem, 可选 GPU
"""
import os
import sys
import time
import signal
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import requests

from workers.common.logging_config import setup_logger, LogContext

# ========== 配置 ==========
# 触发阈值 (边缘用低阈值, 保证高召回)
SIMILARITY_THRESHOLD = float(os.getenv("EDGE_SIMILARITY_THRESHOLD", "0.12"))

# 滑动窗口参数
WINDOW_SIZE_SEC = int(os.getenv("EDGE_WINDOW_SIZE_SEC", "60"))
SLIDE_INTERVAL_SEC = int(os.getenv("EDGE_SLIDE_INTERVAL_SEC", "5"))

# 抽帧参数
FRAME_SAMPLE_FPS = float(os.getenv("EDGE_FRAME_SAMPLE_FPS", "1.0"))
MAX_KEYFRAMES = int(os.getenv("EDGE_MAX_KEYFRAMES", "5"))

# 中心云 API
CENTER_API_URL = os.getenv("CENTER_API_URL", "http://api-gateway:30500")

# 本地 Embedding 服务 (可选, 边缘部署)
LOCAL_EMBEDDING_URL = os.getenv("LOCAL_EMBEDDING_URL", "")

# 模型配置
USE_ONNX = os.getenv("EDGE_USE_ONNX", "false").lower() == "true"
ONNX_MODEL_PATH = os.getenv("EDGE_ONNX_MODEL_PATH", "/models/siglip_int8.onnx")

# 自愈配置
MAX_TRIGGERS_BEFORE_EXIT = int(os.getenv("MAX_TRIGGERS", "1000"))
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "5"))

# 事故模板 (中文, 与中心云保持一致)
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

logger = setup_logger("edge-trigger")
log = LogContext(logger, stage="edge-trigger")


@dataclass
class TriggerEvent:
    """触发事件"""
    camera_id: str
    timestamp: int  # Unix timestamp
    similarity_score: float
    keyframe_count: int
    keyframes_base64: List[str]  # 关键帧 base64 编码
    window_path: str
    trigger_time: str  # ISO 格式时间


@dataclass
class FrameScore:
    """帧分数"""
    frame_idx: int
    timestamp_sec: float
    similarity_score: float
    frame_path: str


class EdgeEmbeddingModel:
    """边缘 Embedding 模型 (支持 ONNX 或 HTTP 调用)"""

    def __init__(self):
        self.template_embeddings: Optional[np.ndarray] = None
        self.onnx_session = None
        self.processor = None

        if USE_ONNX:
            self._load_onnx_model()
        elif LOCAL_EMBEDDING_URL:
            self._load_remote_templates()
        else:
            log.warning("未配置 Embedding 服务, 使用 HTTP 调用中心云")

    def _load_onnx_model(self):
        """加载 ONNX 量化模型 (CPU-only)"""
        try:
            import onnxruntime as ort
            log.info(f"加载 ONNX 模型: {ONNX_MODEL_PATH}")
            self.onnx_session = ort.InferenceSession(
                ONNX_MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            log.info("ONNX 模型加载成功")
            self._compute_template_embeddings_onnx()
        except Exception as e:
            log.error(f"ONNX 模型加载失败: {e}")
            self.onnx_session = None

    def _load_remote_templates(self):
        """从远程 Embedding 服务加载模板嵌入"""
        try:
            log.info(f"从 {LOCAL_EMBEDDING_URL} 加载模板嵌入")
            resp = requests.post(
                f"{LOCAL_EMBEDDING_URL}/encode/text",
                json={"texts": ACCIDENT_TEMPLATES},
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()
            self.template_embeddings = np.array(data["embeddings"])
            log.info(f"模板嵌入加载成功: shape={self.template_embeddings.shape}")
        except Exception as e:
            log.error(f"加载模板嵌入失败: {e}")

    def _compute_template_embeddings_onnx(self):
        """使用 ONNX 计算模板嵌入"""
        if self.onnx_session is None:
            return

        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")

            embeddings = []
            for text in ACCIDENT_TEMPLATES:
                inputs = processor(text=[text], return_tensors="np", padding=True)
                outputs = self.onnx_session.run(
                    ["text_embeds"],
                    {"input_ids": inputs["input_ids"]}
                )
                embeddings.append(outputs[0])

            self.template_embeddings = np.vstack(embeddings)
            # L2 归一化
            self.template_embeddings = self.template_embeddings / np.linalg.norm(
                self.template_embeddings, axis=1, keepdims=True
            )
            log.info(f"ONNX 模板嵌入计算完成: shape={self.template_embeddings.shape}")
        except Exception as e:
            log.error(f"ONNX 模板嵌入计算失败: {e}")

    def encode_images(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """编码图像列表"""
        if USE_ONNX and self.onnx_session:
            return self._encode_images_onnx(image_paths)
        elif LOCAL_EMBEDDING_URL:
            return self._encode_images_remote(image_paths)
        else:
            return None

    def _encode_images_onnx(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """使用 ONNX 编码图像"""
        try:
            from PIL import Image
            from transformers import AutoProcessor

            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-384")

            embeddings = []
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="np")
                outputs = self.onnx_session.run(
                    ["image_embeds"],
                    {"pixel_values": inputs["pixel_values"]}
                )
                embeddings.append(outputs[0])

            result = np.vstack(embeddings)
            # L2 归一化
            result = result / np.linalg.norm(result, axis=1, keepdims=True)
            return result
        except Exception as e:
            log.error(f"ONNX 图像编码失败: {e}")
            return None

    def _encode_images_remote(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """通过远程服务编码图像"""
        try:
            # 读取图像并 base64 编码
            images_b64 = []
            for path in image_paths:
                with open(path, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode())

            resp = requests.post(
                f"{LOCAL_EMBEDDING_URL}/encode/images",
                json={"images": images_b64},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            return np.array(data["embeddings"])
        except Exception as e:
            log.error(f"远程图像编码失败: {e}")
            return None

    def compute_similarity(self, image_embeddings: np.ndarray) -> np.ndarray:
        """计算图像与模板的相似度, 返回每帧的最大相似度"""
        if self.template_embeddings is None:
            log.warning("模板嵌入未加载")
            return np.zeros(len(image_embeddings))

        # 归一化
        img_norm = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )
        tmpl_norm = self.template_embeddings / np.linalg.norm(
            self.template_embeddings, axis=1, keepdims=True
        )

        # 计算相似度矩阵 (N_images x N_templates)
        similarities = np.dot(img_norm, tmpl_norm.T)

        # 每帧取最大相似度
        return similarities.max(axis=1)


class EdgeTrigger:
    """边缘触发器"""

    def __init__(
        self,
        camera_id: str,
        input_dir: str,
        center_api_url: str = CENTER_API_URL
    ):
        self.camera_id = camera_id
        self.input_dir = Path(input_dir) / camera_id
        self.center_api_url = center_api_url.rstrip("/")

        self.trigger_count = 0
        self.running = True
        self.processed_windows = set()

        self.model = EdgeEmbeddingModel()

        # 信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info(f"收到信号 {signum}, 准备退出", camera_id=self.camera_id)
        self.running = False

    def _list_windows(self) -> List[Tuple[int, Path]]:
        """列出滑动窗口文件"""
        if not self.input_dir.exists():
            return []

        windows = []
        for f in self.input_dir.glob("window_*.mp4"):
            if ".tmp" in f.name:
                continue
            try:
                ts_str = f.stem.replace("window_", "")
                ts = int(ts_str)
                windows.append((ts, f))
            except ValueError:
                continue

        windows.sort(key=lambda x: x[0])
        return windows

    def _extract_frames(
        self,
        video_path: Path,
        fps: float = FRAME_SAMPLE_FPS
    ) -> List[str]:
        """从视频提取帧"""
        tmpdir = tempfile.mkdtemp()
        output_pattern = os.path.join(tmpdir, "frame_%04d.jpg")

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
                timeout=60
            )
            if result.returncode != 0:
                log.error(f"FFmpeg 抽帧失败: {video_path}")
                return []
        except Exception as e:
            log.error(f"抽帧异常: {e}")
            return []

        # 收集帧路径
        frames = sorted(Path(tmpdir).glob("frame_*.jpg"))
        return [str(f) for f in frames]

    def _analyze_window(
        self,
        window_path: Path,
        window_ts: int
    ) -> Optional[TriggerEvent]:
        """分析滑动窗口, 判断是否触发"""
        job_id = f"{self.camera_id}_{window_ts}"

        # 1. 抽帧
        frame_paths = self._extract_frames(window_path)
        if not frame_paths:
            log.warning(f"无法抽帧: {window_path}", job_id=job_id)
            return None

        log.info(f"抽取 {len(frame_paths)} 帧", job_id=job_id)

        # 2. 编码图像
        embeddings = self.model.encode_images(frame_paths)
        if embeddings is None or len(embeddings) == 0:
            log.warning(f"图像编码失败", job_id=job_id)
            # 清理帧文件
            for p in frame_paths:
                try:
                    os.remove(p)
                except:
                    pass
            return None

        # 3. 计算相似度
        similarities = self.model.compute_similarity(embeddings)
        max_score = float(similarities.max())
        log.info(f"最高相似度: {max_score:.4f}, 阈值: {SIMILARITY_THRESHOLD}", job_id=job_id)

        # 4. 判断是否触发
        if max_score < SIMILARITY_THRESHOLD:
            log.info(f"未超阈值, 跳过", job_id=job_id)
            # 清理帧文件
            for p in frame_paths:
                try:
                    os.remove(p)
                except:
                    pass
            return None

        # 5. 选择关键帧 (top-k 高分帧)
        top_indices = np.argsort(similarities)[-MAX_KEYFRAMES:]
        keyframe_paths = [frame_paths[i] for i in sorted(top_indices)]

        # 6. 编码关键帧为 base64
        keyframes_b64 = []
        for p in keyframe_paths:
            try:
                with open(p, "rb") as f:
                    keyframes_b64.append(base64.b64encode(f.read()).decode())
            except Exception as e:
                log.warning(f"读取关键帧失败: {e}")

        # 清理帧文件
        for p in frame_paths:
            try:
                os.remove(p)
            except:
                pass

        # 7. 构建触发事件
        return TriggerEvent(
            camera_id=self.camera_id,
            timestamp=window_ts,
            similarity_score=max_score,
            keyframe_count=len(keyframes_b64),
            keyframes_base64=keyframes_b64,
            window_path=str(window_path),
            trigger_time=datetime.utcnow().isoformat() + "Z"
        )

    def _send_trigger_event(self, event: TriggerEvent) -> bool:
        """发送触发事件到中心云"""
        job_id = f"{event.camera_id}_{event.timestamp}"

        try:
            payload = asdict(event)
            resp = requests.post(
                f"{self.center_api_url}/api/v1/edge/events",
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            log.info(f"触发事件已发送: {resp.status_code}", job_id=job_id)
            return True
        except Exception as e:
            log.error(f"发送触发事件失败: {e}", job_id=job_id)
            return False

    def run(self):
        """主循环"""
        log.info(f"启动 Edge Trigger: {self.camera_id}")

        while self.running:
            # 检查自愈阈值
            if self.trigger_count >= MAX_TRIGGERS_BEFORE_EXIT:
                log.info(f"达到自愈阈值 ({MAX_TRIGGERS_BEFORE_EXIT}), 准备退出")
                break

            # 列出窗口
            windows = self._list_windows()

            for ts, window_path in windows:
                if ts in self.processed_windows:
                    continue

                if not self.running:
                    break

                # 分析窗口
                event = self._analyze_window(window_path, ts)

                if event:
                    # 发送触发事件
                    if self._send_trigger_event(event):
                        self.trigger_count += 1
                        log.info(
                            f"触发 {self.trigger_count}/{MAX_TRIGGERS_BEFORE_EXIT}",
                            job_id=f"{self.camera_id}_{ts}"
                        )

                # 标记为已处理 (无论是否触发)
                self.processed_windows.add(ts)

            # 等待下一轮
            time.sleep(POLL_INTERVAL_SEC)

        log.info("Edge Trigger 退出")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Edge Trigger - 边缘事故预检测")
    parser.add_argument("--camera-id", required=True, help="摄像头 ID")
    parser.add_argument(
        "--input-dir",
        default="/data1/videos/windows",
        help="滑动窗口输入目录"
    )
    parser.add_argument(
        "--center-api-url",
        default=CENTER_API_URL,
        help="中心云 API URL"
    )
    args = parser.parse_args()

    trigger = EdgeTrigger(
        camera_id=args.camera_id,
        input_dir=args.input_dir,
        center_api_url=args.center_api_url
    )
    trigger.run()


if __name__ == "__main__":
    main()
