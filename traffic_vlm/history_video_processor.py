"""
历史视频分段处理器

核心功能：
1. 时间段分片（默认5分钟一段）
2. 下载与分析并行执行（生产者-消费者模式）
3. 结果管理（有事故保存证据，无事故清理）
4. SSE进度推送
"""
from __future__ import annotations

import os
import json
import uuid
import shutil
import logging
import threading
import time
from queue import Queue, Empty
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Generator
from enum import Enum

from .tsingcloud_api import TsingcloudAPI, CameraInfo, download_video, TsingcloudAPIError
from .config import HistoryProcessConfig, TsingcloudConfig

logger = logging.getLogger(__name__)


class SegmentStatus(Enum):
    """片段状态"""
    PENDING = "pending"      # 等待中
    DOWNLOADING = "downloading"  # 下载中
    DOWNLOAD_FAILED = "download_failed"  # 下载失败
    ANALYZING = "analyzing"  # 分析中
    ANALYZE_FAILED = "analyze_failed"  # 分析失败
    COMPLETED = "completed"  # 完成
    SKIPPED = "skipped"      # 跳过


class EventType(Enum):
    """SSE事件类型"""
    QUEUE = "queue"          # 队列状态更新
    PROGRESS = "progress"    # 进度更新
    LOG = "log"              # 日志
    RESULT = "result"        # 检出结果
    ERROR = "error"          # 错误
    COMPLETE = "complete"    # 任务完成


@dataclass
class SegmentInfo:
    """片段信息"""
    index: int
    start_time: datetime
    end_time: datetime
    time_range: str  # 显示用，如 "09:00-09:05"

    download_status: SegmentStatus = SegmentStatus.PENDING
    analyze_status: SegmentStatus = SegmentStatus.PENDING
    result: Optional[str] = None  # "detected", "cleared", None

    video_path: Optional[str] = None
    request_id: Optional[str] = None
    retry_count: int = 0
    error_message: Optional[str] = None

    # 分析结果详情
    event_type: Optional[str] = None  # 事故/违法类型
    confidence: float = 0.0
    evidence_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "time_range": self.time_range,
            "download": self.download_status.value,
            "analyze": self.analyze_status.value,
            "result": self.result,
            "retry_count": self.retry_count,
            "error": self.error_message
        }


@dataclass
class TaskInfo:
    """任务信息（支持跨日期时间段）"""
    task_id: str
    road_id: str
    channel_num: str
    start_date: str    # 开始日期，如 "2024-12-17"
    start_time: str    # 开始时间，如 "20:00"
    end_date: str      # 结束日期，如 "2024-12-19"
    end_time: str      # 结束时间，如 "08:00"
    mode: str  # "accident" 或 "violation"
    model: str  # VLM模型
    violation_types: List[str] = field(default_factory=list)

    segments: List[SegmentInfo] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running, completed, stopped

    events_found: int = 0
    events_cleared: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "road_id": self.road_id,
            "channel_num": self.channel_num,
            "start_date": self.start_date,
            "start_time": self.start_time,
            "end_date": self.end_date,
            "end_time": self.end_time,
            "mode": self.mode,
            "model": self.model,
            "total_segments": len(self.segments),
            "status": self.status,
            "events_found": self.events_found,
            "events_cleared": self.events_cleared
        }


class HistoryVideoProcessor:
    """
    历史视频处理器

    实现下载与分析的并行处理：
    - 下载线程（生产者）：顺序下载每个5分钟片段
    - 分析线程（消费者）：分析下载完成的视频
    - 结果根据是否检出事件决定保存或删除
    """

    def __init__(
        self,
        api: TsingcloudAPI,
        config: HistoryProcessConfig,
        pipeline_func: Callable = None,
        event_callback: Callable[[EventType, dict], None] = None
    ):
        """
        初始化处理器

        Args:
            api: 云控智行API客户端
            config: 处理配置
            pipeline_func: 视频分析函数 (video_path, query, mode, model) -> result
            event_callback: SSE事件回调函数
        """
        self.api = api
        self.config = config
        self.pipeline_func = pipeline_func
        self.event_callback = event_callback

        # 任务管理
        self.tasks: Dict[str, TaskInfo] = {}

        # 线程同步
        self.download_queue: Queue = Queue()
        self.analyze_queue: Queue = Queue()
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

    def _emit_event(self, event_type: EventType, data: dict):
        """发送SSE事件"""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"事件回调失败: {e}")

    def _log(self, task_id: str, level: str, message: str, segment: int = None, details: dict = None, category: str = "general"):
        """
        记录日志并发送事件

        Args:
            task_id: 任务ID
            level: 日志级别 (info/warning/error/success/debug)
            message: 日志消息
            segment: 片段索引（可选）
            details: 额外详情（可选）
            category: 日志类别 - "download" | "analyze" | "general"
        """
        logger.info(f"[Task {task_id}] {message}")
        self._emit_event(EventType.LOG, {
            "task_id": task_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "segment": segment,
            "message": message,
            "category": category,
            "details": details or {}
        })

    def _split_time_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        segment_duration: int
    ) -> List[SegmentInfo]:
        """将时间段拆分为多个片段（支持跨日期时间范围）"""
        segments = []
        current = start_dt
        index = 0

        # 判断是否跨日期
        is_cross_date = start_dt.date() != end_dt.date()

        while current < end_dt:
            seg_end = min(current + timedelta(seconds=segment_duration), end_dt)

            # 构建时间范围字符串
            if is_cross_date:
                # 跨日期时包含日期信息
                time_range = f"{current.strftime('%m/%d %H:%M')}-{seg_end.strftime('%H:%M')}"
            else:
                # 同一天只显示时间
                time_range = f"{current.strftime('%H:%M')}-{seg_end.strftime('%H:%M')}"

            segments.append(SegmentInfo(
                index=index,
                start_time=current,
                end_time=seg_end,
                time_range=time_range
            ))

            current = seg_end
            index += 1

        return segments

    def create_task(
        self,
        road_id: str,
        channel_num: str,
        start_date: str,
        start_time: str,
        end_date: str,
        end_time: str,
        mode: str = "accident",
        model: str = "qwen-vl-plus",
        violation_types: List[str] = None,
        segment_duration: int = None
    ) -> TaskInfo:
        """
        创建分析任务（支持跨日期时间段）

        Args:
            road_id: 路口ID
            channel_num: 摄像头通道号
            start_date: 开始日期 "2024-12-17"
            start_time: 开始时间 "20:00"
            end_date: 结束日期 "2024-12-19"
            end_time: 结束时间 "08:00"
            mode: 分析模式 "accident" 或 "violation"
            model: VLM模型
            violation_types: 违法类型列表
            segment_duration: 分片时长（秒）

        Returns:
            TaskInfo 任务信息
        """
        task_id = str(uuid.uuid4())[:8]
        segment_duration = segment_duration or self.config.segment_duration

        # 解析时间（支持跨日期）
        start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")

        # 分片（自动处理跨日期）
        segments = self._split_time_range(start_dt, end_dt, segment_duration)

        # 创建任务
        task = TaskInfo(
            task_id=task_id,
            road_id=road_id,
            channel_num=channel_num,
            start_date=start_date,
            start_time=start_time,
            end_date=end_date,
            end_time=end_time,
            mode=mode,
            model=model,
            violation_types=violation_types or [],
            segments=segments
        )

        # 确保目录存在
        self.config.ensure_dirs()
        task_dir = os.path.join(self.config.result_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        self.tasks[task_id] = task
        return task

    def start_task(self, task_id: str):
        """启动任务（开始下载和分析）"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        # 【修复】启动新任务前，先停止旧任务并清空队列
        self._stop_flag.set()  # 通知可能存在的旧任务停止
        time.sleep(0.3)  # 等待旧线程响应

        # 清空分析队列，防止旧任务的片段混入
        while not self.analyze_queue.empty():
            try:
                self.analyze_queue.get_nowait()
            except Empty:
                break

        self._stop_flag.clear()

        self._log(task_id, "info",
                  f"任务启动 - 路口:{task.road_id}, 摄像头:{task.channel_num}, "
                  f"{task.start_time}-{task.end_time}, 模型:{task.model}")

        # 【修复】立即发送初始队列状态，让前端显示表格
        self._update_queue_status(task)

        # 启动下载线程
        download_thread = threading.Thread(
            target=self._download_worker,
            args=(task,),
            name=f"Download-{task_id}"
        )

        # 启动分析线程
        analyze_thread = threading.Thread(
            target=self._analyze_worker,
            args=(task,),
            name=f"Analyze-{task_id}"
        )

        download_thread.start()
        analyze_thread.start()

        # 等待完成
        download_thread.join()
        self.analyze_queue.put(None)  # 结束信号
        analyze_thread.join()

        # 生成报告
        self._generate_report(task)

        # 发送完成事件
        self._emit_event(EventType.COMPLETE, {
            "task_id": task_id,
            "total_segments": len(task.segments),
            "completed_segments": sum(1 for s in task.segments if s.download_status == SegmentStatus.COMPLETED),
            "skipped_segments": sum(1 for s in task.segments if s.download_status == SegmentStatus.SKIPPED),
            "mode": task.mode,
            "events_found": task.events_found,
            "events_cleared": task.events_cleared,
            "report_url": f"/api/history/report/{task_id}"
        })

        task.status = "completed"

    def stop_task(self, task_id: str):
        """停止任务"""
        self._stop_flag.set()
        task = self.tasks.get(task_id)
        if task:
            task.status = "stopped"
            self._log(task_id, "warning", "任务已停止")

    def retry_segment(self, task_id: str, segment_index: int):
        """重试失败的片段"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if segment_index < 0 or segment_index >= len(task.segments):
            return False

        segment = task.segments[segment_index]
        if segment.download_status == SegmentStatus.DOWNLOAD_FAILED:
            segment.download_status = SegmentStatus.PENDING
            segment.retry_count = 0
            self.download_queue.put(segment)
            self._log(task_id, "info", f"片段#{segment_index} 已加入重试队列", segment_index)
            return True

        return False

    def skip_segment(self, task_id: str, segment_index: int):
        """跳过失败的片段"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if segment_index < 0 or segment_index >= len(task.segments):
            return False

        segment = task.segments[segment_index]
        segment.download_status = SegmentStatus.SKIPPED
        segment.analyze_status = SegmentStatus.SKIPPED
        self._log(task_id, "info", f"片段#{segment_index} 已标记为跳过", segment_index)
        return True

    def _download_worker(self, task: TaskInfo):
        """下载线程：并行下载多个片段（预取模式）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = self.config.max_concurrent_downloads
        logger.info(f"[并行下载] 启用{max_workers}路并行下载")

        # 使用线程池实现预取式并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            segment_iter = iter(task.segments)

            # 初始填充：提交前N个下载任务
            for _ in range(max_workers):
                if self._stop_flag.is_set():
                    break
                seg = next(segment_iter, None)
                if seg:
                    future = executor.submit(self._download_segment_return, task, seg)
                    futures[future] = seg

            # 完成一个，补充一个（流水线模式）
            while futures:
                if self._stop_flag.is_set():
                    # 取消所有未完成的任务
                    for f in futures:
                        f.cancel()
                    break

                # 等待任意一个完成
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # 没有完成的，短暂等待
                    time.sleep(0.1)
                    continue

                for future in done_futures:
                    segment = futures.pop(future)
                    try:
                        success = future.result()
                        if success:
                            logger.info(f"[并行下载] 片段#{segment.index} 完成，放入分析队列")
                    except Exception as e:
                        logger.error(f"[并行下载] 片段#{segment.index} 异常: {e}")

                    # 补充新任务
                    if not self._stop_flag.is_set():
                        next_seg = next(segment_iter, None)
                        if next_seg:
                            new_future = executor.submit(self._download_segment_return, task, next_seg)
                            futures[new_future] = next_seg

        logger.info(f"[并行下载] 下载线程结束")

    def _download_segment_return(self, task: TaskInfo, segment: SegmentInfo) -> bool:
        """下载单个片段并返回结果（用于并行下载）"""
        self._download_segment(task, segment)
        return segment.download_status == SegmentStatus.COMPLETED

    def _download_segment(self, task: TaskInfo, segment: SegmentInfo):
        """下载单个片段"""
        segment.download_status = SegmentStatus.DOWNLOADING

        self._emit_event(EventType.PROGRESS, {
            "task_id": task.task_id,
            "type": "download",
            "segment": segment.index,
            "total": len(task.segments),
            "time_range": segment.time_range,
            "status": "running",
            "message": f"正在下载片段 #{segment.index}"
        })

        self._log(task.task_id, "info",
                  f"片段#{segment.index} 开始下载 ({segment.time_range})",
                  segment.index, category="download")

        retry_count = 0
        max_retries = self.config.download_retry_count

        while retry_count <= max_retries:
            try:
                # 获取视频URL
                video_url = self.api.get_video_url_for_segment(
                    task.road_id,
                    task.channel_num,
                    segment.start_time,
                    segment.end_time,
                    progress_callback=lambda a, m, msg: self._log(
                        task.task_id, "debug", f"轮询 {a}/{m}: {msg}", segment.index, category="download"
                    )
                )

                # 下载视频
                video_filename = f"segment_{segment.index:03d}_{segment.time_range.replace(':', '').replace('-', '_')}.mp4"
                video_path = os.path.join(self.config.temp_dir, task.task_id, video_filename)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)

                # 下载进度回调，保持SSE连接
                def download_progress(downloaded: int, total: int):
                    if total > 0:
                        progress_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        self._log(task.task_id, "debug",
                                  f"下载进度: {progress_mb:.1f}MB / {total_mb:.1f}MB ({downloaded*100//total}%)",
                                  segment.index, category="download")
                    else:
                        # total=0 表示刚开始下载，发送心跳消息防止SSE超时
                        self._log(task.task_id, "debug",
                                  f"正在连接视频服务器...",
                                  segment.index, category="download")

                if download_video(video_url, video_path, progress_callback=download_progress):
                    segment.video_path = video_path
                    segment.download_status = SegmentStatus.COMPLETED

                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    self._log(task.task_id, "success",
                              f"片段#{segment.index} 下载完成 ({file_size:.1f}MB)",
                              segment.index, {"file_size_mb": file_size}, category="download")

                    # 放入分析队列
                    self.analyze_queue.put(segment)
                    self._update_queue_status(task)
                    return

                raise Exception("下载失败")

            except TsingcloudAPIError as e:
                retry_count += 1
                segment.retry_count = retry_count
                segment.error_message = str(e)

                if retry_count <= max_retries:
                    self._log(task.task_id, "warning",
                              f"片段#{segment.index} 下载失败，{self.config.download_retry_interval}秒后重试 ({retry_count}/{max_retries})",
                              segment.index, category="download")
                    self._emit_event(EventType.ERROR, {
                        "task_id": task.task_id,
                        "segment": segment.index,
                        "type": "download",
                        "message": str(e),
                        "retry_count": retry_count,
                        "max_retry": max_retries,
                        "will_retry": True,
                        "retry_in_seconds": self.config.download_retry_interval
                    })
                    time.sleep(self.config.download_retry_interval)
                else:
                    break

            except Exception as e:
                retry_count += 1
                segment.error_message = str(e)
                if retry_count <= max_retries:
                    time.sleep(self.config.download_retry_interval)
                else:
                    break

        # 所有重试失败
        segment.download_status = SegmentStatus.DOWNLOAD_FAILED
        self._log(task.task_id, "error",
                  f"片段#{segment.index} 下载失败（已重试{max_retries}次）: {segment.error_message}",
                  segment.index, category="download")
        self._emit_event(EventType.ERROR, {
            "task_id": task.task_id,
            "segment": segment.index,
            "type": "download",
            "message": segment.error_message,
            "retry_count": max_retries,
            "max_retry": max_retries,
            "will_retry": False
        })
        self._update_queue_status(task)

    def _analyze_worker(self, task: TaskInfo):
        """分析线程：消费下载完成的视频"""
        while True:
            try:
                segment = self.analyze_queue.get(timeout=2)  # 减少等待间隙，防止SSE超时
                if segment is None:  # 结束信号
                    break
                self._analyze_segment(task, segment)
            except Empty:
                if self._stop_flag.is_set():
                    break
                continue

    def _analyze_segment(self, task: TaskInfo, segment: SegmentInfo):
        """分析单个片段"""
        if not segment.video_path or not os.path.exists(segment.video_path):
            segment.analyze_status = SegmentStatus.ANALYZE_FAILED
            segment.error_message = "视频文件不存在"
            return

        segment.analyze_status = SegmentStatus.ANALYZING

        self._emit_event(EventType.PROGRESS, {
            "task_id": task.task_id,
            "type": "analyze",
            "segment": segment.index,
            "total": len(task.segments),
            "time_range": segment.time_range,
            "status": "running",
            "message": f"正在分析片段 #{segment.index}"
        })

        self._log(task.task_id, "info",
                  f"片段#{segment.index} 开始分析",
                  segment.index, category="analyze")

        try:
            # 构建查询词
            if task.mode == "accident":
                user_query = "交通事故"
            else:
                user_query = "交通违法: " + ", ".join(task.violation_types) if task.violation_types else "交通违法"

            # 创建进度回调 - 将 Pipeline 内部进度发送到 SSE
            def progress_callback(percent: int, message: str):
                self._log(task.task_id, "debug",
                          f"[{percent}%] {message}",
                          segment.index, category="analyze")

            # 调用分析pipeline
            has_event = False
            event_type = None
            confidence = 0.0
            analysis_result = {}

            if self.pipeline_func:
                analysis_result = self.pipeline_func(
                    video_path=segment.video_path,
                    user_query=user_query,
                    mode=task.mode,
                    model=task.model,
                    progress_callback=progress_callback  # 传入进度回调
                )

                # 解析结果
                has_event = analysis_result.get("has_event", False)
                event_type = analysis_result.get("event_type")
                confidence = analysis_result.get("confidence", 0.0)
            else:
                # 没有pipeline，模拟分析
                logger.warning("未配置分析pipeline，跳过实际分析")
                time.sleep(2)  # 模拟分析耗时

            segment.analyze_status = SegmentStatus.COMPLETED

            if has_event:
                # 检出事件 - 保存证据
                segment.result = "detected"
                segment.event_type = event_type
                segment.confidence = confidence
                task.events_found += 1

                # 保存证据
                evidence_path = self._save_evidence(task, segment, analysis_result)
                segment.evidence_path = evidence_path

                self._log(task.task_id, "warning",
                          f"⚠️ 片段#{segment.index} 检测到{event_type} (置信度:{confidence:.2f})",
                          segment.index, category="analyze")

                # 发送结果事件
                self._emit_event(EventType.RESULT, {
                    "task_id": task.task_id,
                    "segment": segment.index,
                    "time": segment.start_time.strftime("%H:%M:%S"),
                    "mode": task.mode,
                    "event_type": event_type,
                    "confidence": confidence,
                    "thumbnail": f"/api/history/thumbnail/{task.task_id}/{segment.index}"
                })

            else:
                # 无事件 - 清理视频
                segment.result = "cleared"
                task.events_cleared += 1

                if self.config.cleanup_on_no_event:
                    self._cleanup_segment(segment)
                    self._log(task.task_id, "success",
                              f"片段#{segment.index} 分析完成 - 无事故 - 已清理 ✓",
                              segment.index, {"action": "cleanup"}, category="analyze")
                else:
                    self._log(task.task_id, "success",
                              f"片段#{segment.index} 分析完成 - 无事故",
                              segment.index, category="analyze")

        except Exception as e:
            segment.analyze_status = SegmentStatus.ANALYZE_FAILED
            segment.error_message = str(e)
            self._log(task.task_id, "error",
                      f"片段#{segment.index} 分析失败: {e}",
                      segment.index, category="analyze")

        self._update_queue_status(task)

    def _save_evidence(self, task: TaskInfo, segment: SegmentInfo, analysis_result: dict) -> str:
        """保存事件证据"""
        task_dir = os.path.join(self.config.result_dir, task.task_id)
        segment_dir = os.path.join(task_dir, f"segment_{segment.index:03d}")
        os.makedirs(segment_dir, exist_ok=True)

        # 移动原始视频
        if segment.video_path and os.path.exists(segment.video_path):
            original_path = os.path.join(segment_dir, "original.mp4")
            shutil.move(segment.video_path, original_path)
            segment.video_path = original_path

        # 保存分析结果
        result_path = os.path.join(segment_dir, "vlm_result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump({
                "segment_index": segment.index,
                "time_range": segment.time_range,
                "event_type": segment.event_type,
                "confidence": segment.confidence,
                "analysis": analysis_result
            }, f, ensure_ascii=False, indent=2)

        # 保存关键帧（如果有）
        keyframes = analysis_result.get("keyframes", [])
        if keyframes:
            keyframes_dir = os.path.join(segment_dir, "keyframes")
            os.makedirs(keyframes_dir, exist_ok=True)
            for i, kf in enumerate(keyframes):
                if isinstance(kf, str) and os.path.exists(kf):
                    dst = os.path.join(keyframes_dir, f"frame_{i:03d}.jpg")
                    shutil.copy(kf, dst)

        return segment_dir

    def _cleanup_segment(self, segment: SegmentInfo):
        """清理无事件的片段"""
        if segment.video_path and os.path.exists(segment.video_path):
            try:
                os.remove(segment.video_path)
                segment.video_path = None
            except Exception as e:
                logger.warning(f"清理视频失败: {e}")

    def _update_queue_status(self, task: TaskInfo):
        """更新并发送队列状态"""
        completed = sum(1 for s in task.segments
                       if s.download_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED]
                       and s.analyze_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED])

        self._emit_event(EventType.QUEUE, {
            "task_id": task.task_id,
            "segments": [s.to_dict() for s in task.segments],
            "completed": completed,
            "total": len(task.segments)
        })

    def _generate_report(self, task: TaskInfo):
        """生成HTML报告"""
        task_dir = os.path.join(self.config.result_dir, task.task_id)
        report_path = os.path.join(task_dir, "report.html")

        # 收集事件
        events = []
        for seg in task.segments:
            if seg.result == "detected":
                events.append({
                    "index": seg.index,
                    "time_range": seg.time_range,
                    "event_type": seg.event_type,
                    "confidence": seg.confidence,
                    "evidence_path": seg.evidence_path
                })

        # 生成简单的HTML报告
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>历史视频分析报告 - {task.task_id}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .event-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        .event-card.detected {{ border-left: 4px solid #ff9800; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; }}
        .badge-warning {{ background: #fff3e0; color: #e65100; }}
        .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 历史视频分析报告</h1>

        <div class="summary">
            <p><strong>任务ID:</strong> {task.task_id}</p>
            <p><strong>路口:</strong> {task.road_id} | <strong>摄像头:</strong> {task.channel_num}</p>
            <p><strong>时间段:</strong> {task.start_date} {task.start_time} → {task.end_date} {task.end_time}</p>
            <p><strong>分析模式:</strong> {"交通事故检测" if task.mode == "accident" else "交通违法检测"}</p>
            <p><strong>VLM模型:</strong> {task.model}</p>
            <hr>
            <p><strong>总片段数:</strong> {len(task.segments)}</p>
            <p><strong>检出事件:</strong> <span class="badge badge-warning">{task.events_found} 起</span></p>
            <p><strong>无异常:</strong> <span class="badge badge-success">{task.events_cleared} 个</span></p>
        </div>

        <h2>📋 检出事件详情</h2>
        {"".join(f'''
        <div class="event-card detected">
            <h3>#{e["index"]} - {e["event_type"]}</h3>
            <p>时间段: {e["time_range"]}</p>
            <p>置信度: {e["confidence"]:.2%}</p>
            <p>证据路径: {e["evidence_path"]}</p>
        </div>
        ''' for e in events) if events else '<p>未检出任何事件</p>'}
    </div>
</body>
</html>"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self._log(task.task_id, "info", f"报告已生成: {report_path}")
        return report_path

    def get_task_status(self, task_id: str) -> Optional[dict]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        completed = sum(1 for s in task.segments
                       if s.analyze_status == SegmentStatus.COMPLETED)

        return {
            **task.to_dict(),
            "completed_segments": completed,
            "segments": [s.to_dict() for s in task.segments]
        }

    def stream_progress(self, task_id: str) -> Generator[str, None, None]:
        """SSE进度流生成器"""
        task = self.tasks.get(task_id)
        if not task:
            yield f"event: error\ndata: {{\"message\": \"任务不存在\"}}\n\n"
            return

        # 初始状态
        yield f"event: queue\ndata: {json.dumps({'segments': [s.to_dict() for s in task.segments], 'completed': 0, 'total': len(task.segments)})}\n\n"

        # 持续推送直到任务完成
        while task.status == "running":
            time.sleep(1)
            # 实际事件通过event_callback推送

        # 最终状态
        yield f"event: complete\ndata: {json.dumps(task.to_dict())}\n\n"
