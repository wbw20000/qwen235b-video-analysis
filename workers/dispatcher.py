#!/usr/bin/env python3
"""
Dispatcher Worker - 阶段1.5 核心组件
- 监听 seg_*.mp4 完成文件（忽略 .tmp）
- 生成重叠窗口: window[i] = tail(seg[i-1], 30s) + seg[i]
- 使用 TS 中间格式 concat 再 remux 回 MP4
- 生成确定性 job_id = {camera_id}_{seg_end_ts}
- XADD video_tasks
"""
import os
import sys
import time
import signal
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, VideoTask

# 配置
SEGMENT_DURATION_SEC = 600  # 10 分钟
OVERLAP_DURATION_SEC = 30   # 30 秒重叠
POLL_INTERVAL_SEC = 5       # 轮询间隔
MAX_TASKS_BEFORE_EXIT = 100 # 自愈阈值


class Dispatcher:
    """分片调度器 - 生成重叠窗口"""

    def __init__(
        self,
        camera_id: str,
        input_dir: str,
        output_dir: str,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        overlap_sec: int = OVERLAP_DURATION_SEC
    ):
        self.camera_id = camera_id
        self.input_dir = Path(input_dir) / camera_id
        self.output_dir = Path(output_dir) / camera_id
        self.overlap_sec = overlap_sec
        self.task_count = 0
        self.running = True
        self.processed_segments = set()  # 已处理的分片

        self.logger = setup_logger("dispatcher")
        self.log = LogContext(self.logger, camera_id=camera_id, stage="dispatch")
        self.redis = RedisStreamClient(redis_host, redis_port)

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.log.info(f"收到信号 {signum}，准备退出")
        self.running = False

    def _list_completed_segments(self) -> list:
        """
        列出已完成的分片文件（非 .tmp）
        返回按时间戳排序的列表
        """
        if not self.input_dir.exists():
            return []

        segments = []
        for f in self.input_dir.glob("seg_*.mp4"):
            # 跳过临时文件
            if f.suffix == ".tmp" or ".tmp" in f.name:
                continue

            # 解析时间戳
            try:
                ts_str = f.stem.replace("seg_", "")
                ts = int(ts_str)
                segments.append((ts, f))
            except ValueError:
                continue

        # 按时间戳排序
        segments.sort(key=lambda x: x[0])
        return segments

    def _get_video_duration(self, video_path: Path) -> float:
        """获取视频时长（秒）"""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            if result.returncode == 0:
                return float(result.stdout.decode().strip())
        except Exception as e:
            self.log.error(f"获取视频时长失败: {e}")
        return 0.0

    def _extract_tail(self, video_path: Path, tail_sec: int, output_ts: Path) -> bool:
        """
        从视频末尾提取指定秒数，输出为 TS 格式
        """
        duration = self._get_video_duration(video_path)
        if duration <= 0:
            self.log.error(f"无法获取视频时长: {video_path}")
            return False

        start_time = max(0, duration - tail_sec)

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-c", "copy",
            "-f", "mpegts",
            str(output_ts)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
            return result.returncode == 0
        except Exception as e:
            self.log.error(f"提取尾部失败: {e}")
            return False

    def _convert_to_ts(self, video_path: Path, output_ts: Path) -> bool:
        """将视频转换为 TS 格式"""
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-c", "copy",
            "-f", "mpegts",
            str(output_ts)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            self.log.error(f"转换 TS 失败: {e}")
            return False

    def _concat_ts_to_mp4(self, ts_files: list, output_mp4: Path) -> bool:
        """
        合并多个 TS 文件为 MP4
        使用 concat 协议
        """
        # 构建 concat 输入
        concat_str = "|".join(str(f) for f in ts_files)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", f"concat:{concat_str}",
            "-c", "copy",
            "-f", "mp4",
            "-movflags", "+faststart",
            str(output_mp4)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180
            )
            return result.returncode == 0
        except Exception as e:
            self.log.error(f"合并 MP4 失败: {e}")
            return False

    def _create_window(
        self,
        prev_segment: Optional[Path],
        curr_segment: Path,
        seg_end_ts: int
    ) -> Optional[Path]:
        """
        创建重叠窗口
        window[i] = tail(seg[i-1], 30s) + seg[i]

        如果没有前一个分片，则直接使用当前分片
        """
        # 确定性 job_id
        job_id = f"{self.camera_id}_{seg_end_ts}"
        window_filename = f"window_{seg_end_ts}.mp4"
        tmp_path = self.output_dir / f"{window_filename}.tmp"
        final_path = self.output_dir / window_filename

        # 检查是否已存在（幂等性）
        if final_path.exists():
            self.log.info(f"窗口已存在，跳过: {final_path.name}", job_id=job_id)
            return final_path

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            ts_files = []

            # 1. 提取前一个分片的尾部（如果存在）
            if prev_segment and prev_segment.exists():
                prev_tail_ts = tmpdir / "prev_tail.ts"
                if self._extract_tail(prev_segment, self.overlap_sec, prev_tail_ts):
                    ts_files.append(prev_tail_ts)
                    self.log.debug(f"提取尾部成功: {prev_segment.name}")
                else:
                    self.log.warning(f"提取尾部失败，跳过重叠: {prev_segment.name}")

            # 2. 转换当前分片为 TS
            curr_ts = tmpdir / "curr.ts"
            if not self._convert_to_ts(curr_segment, curr_ts):
                self.log.error(f"转换 TS 失败: {curr_segment.name}")
                return None
            ts_files.append(curr_ts)

            # 3. 合并为 MP4
            if len(ts_files) == 1:
                # 没有重叠，直接复制
                import shutil
                shutil.copy(curr_segment, tmp_path)
            else:
                # 合并
                if not self._concat_ts_to_mp4(ts_files, tmp_path):
                    self.log.error(f"合并失败: {window_filename}")
                    return None

        # 4. 原子重命名
        try:
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.rename(tmp_path, final_path)
            self.log.info(f"窗口创建成功: {final_path.name}", job_id=job_id)
            return final_path
        except Exception as e:
            self.log.error(f"重命名失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return None

    def _submit_task(self, window_path: Path, seg_end_ts: int) -> bool:
        """提交任务到 Redis Stream"""
        job_id = f"{self.camera_id}_{seg_end_ts}"
        trace_id = job_id  # 使用 job_id 作为 trace_id

        task = VideoTask(
            job_id=job_id,
            camera_id=self.camera_id,
            window_path=str(window_path),
            seg_end_ts=seg_end_ts,
            trace_id=trace_id
        )

        try:
            self.redis.add_video_task(task)
            self.log.info(f"任务已提交: {job_id}", job_id=job_id, trace_id=trace_id)
            return True
        except Exception as e:
            self.log.error(f"提交任务失败: {e}", job_id=job_id)
            return False

    def run(self):
        """主循环"""
        self.log.info(f"启动 Dispatcher: {self.camera_id}")

        while self.running:
            # 检查自愈阈值
            if self.task_count >= MAX_TASKS_BEFORE_EXIT:
                self.log.info(f"达到自愈阈值 ({MAX_TASKS_BEFORE_EXIT})，准备退出")
                break

            # 列出已完成的分片
            segments = self._list_completed_segments()

            # 处理未处理的分片
            for i, (ts, seg_path) in enumerate(segments):
                if ts in self.processed_segments:
                    continue

                if not self.running:
                    break

                # 获取前一个分片
                prev_segment = None
                if i > 0:
                    _, prev_segment = segments[i - 1]

                # 创建窗口
                window_path = self._create_window(prev_segment, seg_path, ts)
                if window_path:
                    # 提交任务
                    if self._submit_task(window_path, ts):
                        self.processed_segments.add(ts)
                        self.task_count += 1
                        self.log.info(
                            f"已处理 {self.task_count}/{MAX_TASKS_BEFORE_EXIT} 个分片",
                            job_id=f"{self.camera_id}_{ts}"
                        )

            # 等待下一轮
            time.sleep(POLL_INTERVAL_SEC)

        self.log.info("Dispatcher 退出")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dispatcher Worker")
    parser.add_argument("--camera-id", required=True, help="摄像头 ID")
    parser.add_argument(
        "--input-dir",
        default="/data1/videos/rtsp_recordings",
        help="分片输入目录"
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/videos/windows",
        help="窗口输出目录"
    )
    parser.add_argument("--redis-host", default="localhost", help="Redis 主机")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口")
    parser.add_argument(
        "--overlap-sec",
        type=int,
        default=OVERLAP_DURATION_SEC,
        help="重叠时长（秒）"
    )
    args = parser.parse_args()

    dispatcher = Dispatcher(
        camera_id=args.camera_id,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        overlap_sec=args.overlap_sec
    )
    dispatcher.run()


if __name__ == "__main__":
    main()
