#!/usr/bin/env python3
"""
RTSP Ingest Worker
- 从 RTSP 流录制 10 分钟固定分片
- 使用 ffmpeg remux（不转码）
- 原子写入：先写 .tmp，完成后 rename 为 .mp4
- 自愈：每 N 个分片后退出让 K8S 重启
"""
import os
import sys
import time
import signal
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.common.logging_config import setup_logger, LogContext

# 配置
SEGMENT_DURATION_SEC = 600  # 10 分钟
MAX_SEGMENTS_BEFORE_EXIT = 50  # 自愈：处理 50 个分片后退出
RECONNECT_DELAY_SEC = 5


class RTSPIngest:
    """RTSP 录制器"""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        output_dir: str,
        segment_duration: int = SEGMENT_DURATION_SEC
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir) / camera_id
        self.segment_duration = segment_duration
        self.segment_count = 0
        self.running = True
        self.logger = setup_logger("rtsp-ingest")
        self.log = LogContext(self.logger, camera_id=camera_id, stage="ingest")

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.log.info(f"收到信号 {signum}，准备退出")
        self.running = False

    def _generate_segment_path(self) -> tuple:
        """生成分片文件路径"""
        end_ts = int(time.time())
        filename = f"seg_{end_ts}.mp4"
        tmp_path = self.output_dir / f"{filename}.tmp"
        final_path = self.output_dir / filename
        return tmp_path, final_path, end_ts

    def _record_segment(self, tmp_path: Path) -> bool:
        """
        录制一个分片
        使用 ffmpeg remux（-c copy）不转码
        """
        cmd = [
            "ffmpeg",
            "-y",  # 覆盖已存在文件
            "-rtsp_transport", "tcp",  # 使用 TCP 传输
            "-i", self.rtsp_url,
            "-t", str(self.segment_duration),  # 录制时长
            "-c", "copy",  # remux 不转码
            "-f", "mp4",
            "-movflags", "+faststart",
            str(tmp_path)
        ]

        self.log.info(f"开始录制分片: {tmp_path.name}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.segment_duration + 60  # 超时保护
            )
            if result.returncode == 0:
                return True
            else:
                self.log.error(
                    f"ffmpeg 录制失败: {result.stderr.decode()[-500:]}"
                )
                return False
        except subprocess.TimeoutExpired:
            self.log.error("ffmpeg 录制超时")
            return False
        except Exception as e:
            self.log.error(f"录制异常: {e}")
            return False

    def _atomic_rename(self, tmp_path: Path, final_path: Path) -> bool:
        """
        原子重命名：确保文件完整后才可见
        先 fsync 确保数据落盘，再 rename
        """
        try:
            # 确保数据落盘
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())

            # 原子重命名
            os.rename(tmp_path, final_path)
            self.log.info(f"分片完成: {final_path.name}")
            return True
        except Exception as e:
            self.log.error(f"重命名失败: {e}")
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
            return False

    def run(self):
        """主循环"""
        self.log.info(f"启动 RTSP 录制: {self.rtsp_url}")

        while self.running:
            # 检查是否需要自愈退出
            if self.segment_count >= MAX_SEGMENTS_BEFORE_EXIT:
                self.log.info(
                    f"达到自愈阈值 ({MAX_SEGMENTS_BEFORE_EXIT} 分片)，准备退出"
                )
                break

            # 生成文件路径
            tmp_path, final_path, end_ts = self._generate_segment_path()

            # 跳过已存在的分片
            if final_path.exists():
                self.log.info(f"分片已存在，跳过: {final_path.name}")
                time.sleep(1)
                continue

            # 录制分片
            if self._record_segment(tmp_path):
                # 原子重命名
                if self._atomic_rename(tmp_path, final_path):
                    self.segment_count += 1
                    self.log.info(
                        f"已完成 {self.segment_count}/{MAX_SEGMENTS_BEFORE_EXIT} 分片"
                    )
            else:
                # 录制失败，清理临时文件并重试
                if tmp_path.exists():
                    tmp_path.unlink()
                self.log.warning(f"录制失败，{RECONNECT_DELAY_SEC}s 后重试")
                time.sleep(RECONNECT_DELAY_SEC)

        self.log.info("RTSP 录制器退出")


def main():
    parser = argparse.ArgumentParser(description="RTSP Ingest Worker")
    parser.add_argument("--camera-id", required=True, help="摄像头 ID")
    parser.add_argument("--rtsp-url", required=True, help="RTSP 流地址")
    parser.add_argument(
        "--output-dir",
        default="/data1/videos/rtsp_recordings",
        help="输出目录"
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=SEGMENT_DURATION_SEC,
        help="分片时长（秒）"
    )
    args = parser.parse_args()

    ingest = RTSPIngest(
        camera_id=args.camera_id,
        rtsp_url=args.rtsp_url,
        output_dir=args.output_dir,
        segment_duration=args.segment_duration
    )
    ingest.run()


if __name__ == "__main__":
    main()
