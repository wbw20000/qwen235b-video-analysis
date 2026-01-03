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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Generator
from enum import Enum

from .tsingcloud_api import TsingcloudAPI, CameraInfo, download_video, download_video_rtsp, TsingcloudAPIError, DualAccountDownloader
from .config import HistoryProcessConfig, TsingcloudConfig, TrafficVLMConfig

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

    # 多摄像头模式字段
    camera_index: Optional[int] = None    # 来自哪个摄像头（索引）
    camera_channel: Optional[str] = None  # 来自哪个摄像头（通道号）

    def to_dict(self) -> dict:
        result = {
            "index": self.index,
            "time_range": self.time_range,
            "download": self.download_status.value,
            "analyze": self.analyze_status.value,
            "result": self.result,
            "retry_count": self.retry_count,
            "error": self.error_message
        }
        if self.camera_index is not None:
            result["camera_index"] = self.camera_index
            result["camera_channel"] = self.camera_channel
        return result


@dataclass
class CameraTaskInfo:
    """单个摄像头的任务信息（用于多摄像头模式）"""
    index: int                    # 摄像头索引
    channel_num: str              # 通道编号（HTTP接口返回）
    device_id: str = ""           # 设备ID（从channel_num反查）

    segments: List[SegmentInfo] = field(default_factory=list)
    status: str = "pending"       # pending, running, completed, failed
    completed_count: int = 0
    failed_count: int = 0
    events_found: int = 0
    events_cleared: int = 0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "channel_num": self.channel_num,
            "device_id": self.device_id,
            "status": self.status,
            "total_segments": len(self.segments),
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "events_found": self.events_found,
            "events_cleared": self.events_cleared,
            "segments": [s.to_dict() for s in self.segments]
        }


@dataclass
class TaskInfo:
    """任务信息（支持跨日期时间段和多摄像头）"""
    task_id: str
    road_id: str
    channel_num: str             # 单摄像头时使用（兼容旧逻辑）
    start_date: str    # 开始日期，如 "2024-12-17"
    start_time: str    # 开始时间，如 "20:00"
    end_date: str      # 结束日期，如 "2024-12-19"
    end_time: str      # 结束时间，如 "08:00"
    mode: str  # "accident" 或 "violation"
    model: str  # VLM模型
    violation_types: List[str] = field(default_factory=list)

    # 下载方式配置
    download_method: str = "auto"  # "auto"(先RTSP后HTTP), "rtsp", "http"
    device_cate: str = "DJ"        # RTSP设备类型: "DJ"(全景) 或 "KK"(抓拍)
    device_index: int = 0          # RTSP设备索引（同类型中第几个）

    # 多摄像头支持
    channel_nums: List[str] = field(default_factory=list)  # 多摄像头通道列表
    camera_tasks: List[CameraTaskInfo] = field(default_factory=list)  # 每个摄像头的子任务

    segments: List[SegmentInfo] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"  # running, completed, stopped

    events_found: int = 0
    events_cleared: int = 0

    def is_multi_camera(self) -> bool:
        """是否为多摄像头模式"""
        return len(self.channel_nums) > 1 or len(self.camera_tasks) > 0

    def to_dict(self) -> dict:
        result = {
            "task_id": self.task_id,
            "road_id": self.road_id,
            "channel_num": self.channel_num,
            "start_date": self.start_date,
            "start_time": self.start_time,
            "end_date": self.end_date,
            "end_time": self.end_time,
            "mode": self.mode,
            "model": self.model,
            "download_method": self.download_method,
            "total_segments": len(self.segments),
            "status": self.status,
            "events_found": self.events_found,
            "events_cleared": self.events_cleared
        }
        # 多摄像头模式额外字段
        if self.channel_nums:
            result["channel_nums"] = self.channel_nums
        if self.camera_tasks:
            result["camera_tasks"] = [ct.to_dict() for ct in self.camera_tasks]
            result["total_cameras"] = len(self.camera_tasks)
        return result


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
        event_callback: Callable[[EventType, dict], None] = None,
        tsingcloud_config: TsingcloudConfig = None
    ):
        """
        初始化处理器

        Args:
            api: 云控智行API客户端
            config: 处理配置
            pipeline_func: 视频分析函数 (video_path, query, mode, model) -> result
            event_callback: SSE事件回调函数
            tsingcloud_config: 云控API配置（用于检查RTSP权限等）
        """
        self.api = api
        self.config = config
        self.pipeline_func = pipeline_func
        self.event_callback = event_callback
        self.tsingcloud_config = tsingcloud_config or TsingcloudConfig()

        # 双账号下载器（先RTSP后HTTP轮询）
        self.dual_downloader: Optional[DualAccountDownloader] = None
        # 设备映射器缓存（用于缓存路径生成，所有模式共用）
        self._cached_device_mapper = None
        if self.tsingcloud_config.enable_rtsp:
            try:
                self.dual_downloader = DualAccountDownloader(
                    http_app_key=self.tsingcloud_config.app_key,
                    http_password=self.tsingcloud_config.password,
                    rtsp_app_key=self.tsingcloud_config.rtsp_app_key,
                    rtsp_password=self.tsingcloud_config.rtsp_password,
                    device_mapping_file=self.tsingcloud_config.device_mapping_file,
                    base_url=self.tsingcloud_config.base_url,
                    poll_interval=self.tsingcloud_config.poll_interval,
                    poll_timeout=self.tsingcloud_config.poll_timeout,
                    rtsp_download_timeout=self.config.get_rtsp_download_timeout()
                )
                logger.info(f"双账号下载器已初始化（先RTSP后HTTP轮询，RTSP超时={self.config.get_rtsp_download_timeout()}秒）")
            except Exception as e:
                logger.warning(f"双账号下载器初始化失败: {e}，将使用HTTP轮询模式")
                self.dual_downloader = None

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
        channel_num: str = "",
        start_date: str = "",
        start_time: str = "",
        end_date: str = "",
        end_time: str = "",
        mode: str = "accident",
        model: str = "qwen-vl-plus",
        violation_types: List[str] = None,
        segment_duration: int = None,
        download_method: str = "auto",
        device_cate: str = "DJ",
        device_index: int = 0,
        channel_nums: List[str] = None
    ) -> TaskInfo:
        """
        创建分析任务（支持跨日期时间段和多摄像头）

        Args:
            road_id: 路口ID
            channel_num: 摄像头通道号（单摄像头模式，HTTP轮询使用）
            start_date: 开始日期 "2024-12-17"
            start_time: 开始时间 "20:00"
            end_date: 结束日期 "2024-12-19"
            end_time: 结束时间 "08:00"
            mode: 分析模式 "accident" 或 "violation"
            model: VLM模型
            violation_types: 违法类型列表
            segment_duration: 分片时长（秒）
            download_method: 下载方式 "auto"(先RTSP后HTTP), "rtsp", "http"
            device_cate: RTSP设备类型 "DJ"(全景) 或 "KK"(抓拍)
            device_index: RTSP设备索引（同类型中第几个）
            channel_nums: 多摄像头通道号列表（多摄像头模式）

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

        # 处理多摄像头：兼容新旧参数
        if channel_nums is None:
            channel_nums = []
        if channel_num and not channel_nums:
            channel_nums = [channel_num]  # 单摄像头转为列表
        if not channel_num and channel_nums:
            channel_num = channel_nums[0]  # 保持兼容，取第一个

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
            download_method=download_method,
            device_cate=device_cate,
            device_index=device_index,
            channel_nums=channel_nums,
            segments=segments
        )

        # 多摄像头模式：为每个摄像头创建 CameraTaskInfo
        if len(channel_nums) > 1:
            for i, ch_num in enumerate(channel_nums):
                device_id = self._get_device_id_by_channel(road_id, ch_num, device_cate)
                # 为每个摄像头克隆segments
                camera_segments = self._split_time_range(start_dt, end_dt, segment_duration)
                camera_task = CameraTaskInfo(
                    index=i,
                    channel_num=ch_num,
                    device_id=device_id,
                    segments=camera_segments,
                    status="pending"
                )
                task.camera_tasks.append(camera_task)
            # 多摄像头模式下，task.segments 为空（使用 camera_tasks）
            task.segments = []

        # 确保目录存在
        self.config.ensure_dirs()
        task_dir = os.path.join(self.config.result_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        self.tasks[task_id] = task
        return task

    def _get_device_id_by_channel(self, road_id: str, channel_num: str, device_cate: str = "DJ") -> str:
        """
        根据channel_num反查设备ID（用于缓存路径）

        Args:
            road_id: 路口ID
            channel_num: 通道编号（HTTP接口返回）
            device_cate: 设备类型

        Returns:
            设备ID，未找到返回空字符串
        """
        try:
            # 获取可用的 device_mapper（复用缓存或创建新实例）
            device_mapper = None
            if self.dual_downloader and self.dual_downloader.device_mapper:
                device_mapper = self.dual_downloader.device_mapper
            elif self._cached_device_mapper:
                device_mapper = self._cached_device_mapper
            elif self.tsingcloud_config and self.tsingcloud_config.device_mapping_file:
                from .tsingcloud_api import DeviceMapper
                self._cached_device_mapper = DeviceMapper(self.tsingcloud_config.device_mapping_file)
                device_mapper = self._cached_device_mapper

            if device_mapper:
                all_devices = device_mapper.get_all_devices(road_id, device_cate)
                for device in all_devices:
                    sn = device.get('sn', '')
                    # SN格式：2_xxx_channel_num，channel_num在末尾
                    if sn.endswith(channel_num) or channel_num in sn:
                        return device.get('deviceId', '')
        except Exception as e:
            logger.warning(f"反查设备ID失败: {e}")
        return ""

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

        # 日志：下载方式配置
        method_names = {"auto": "自动(先RTSP后HTTP)", "rtsp": "仅RTSP", "http": "仅HTTP轮询"}
        dl_method_str = method_names.get(task.download_method, task.download_method)
        dual_status = "已启用" if self.dual_downloader else "未启用"

        # 多摄像头模式日志
        if task.is_multi_camera():
            camera_count = len(task.camera_tasks)
            total_segments = sum(len(ct.segments) for ct in task.camera_tasks)
            self._log(task_id, "info",
                      f"任务启动 - 路口:{task.road_id}, 多摄像头模式({camera_count}个), "
                      f"{task.start_time}-{task.end_time}, 模型:{task.model}")
            self._log(task_id, "info",
                      f"下载方式: {dl_method_str}, 总片段数: {total_segments}")
        else:
            self._log(task_id, "info",
                      f"任务启动 - 路口:{task.road_id}, 摄像头:{task.channel_num}, "
                      f"{task.start_time}-{task.end_time}, 模型:{task.model}")
            self._log(task_id, "info",
                      f"下载方式: {dl_method_str}, 双账号下载器: {dual_status}, "
                      f"设备类型: {task.device_cate}, 索引: {task.device_index}")

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

        # 发送完成事件（兼容多摄像头模式）
        if task.is_multi_camera():
            total_segments = sum(len(ct.segments) for ct in task.camera_tasks)
            completed_segments = sum(ct.completed_count for ct in task.camera_tasks)
            skipped_segments = sum(1 for ct in task.camera_tasks
                                   for s in ct.segments if s.download_status == SegmentStatus.SKIPPED)
        else:
            total_segments = len(task.segments)
            completed_segments = sum(1 for s in task.segments if s.download_status == SegmentStatus.COMPLETED)
            skipped_segments = sum(1 for s in task.segments if s.download_status == SegmentStatus.SKIPPED)

        self._emit_event(EventType.COMPLETE, {
            "task_id": task_id,
            "total_segments": total_segments,
            "completed_segments": completed_segments,
            "skipped_segments": skipped_segments,
            "mode": task.mode,
            "events_found": task.events_found,
            "events_cleared": task.events_cleared,
            "report_url": f"/api/history/report/{task_id}",
            "multi_camera": task.is_multi_camera(),
            "camera_count": len(task.camera_tasks) if task.is_multi_camera() else 1
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
        """下载线程：支持单摄像头和多摄像头模式

        - 单摄像头模式：限制并发下载同一摄像头的片段（默认2），避免API限流
        - 多摄像头模式：摄像头间并行，每个摄像头内部限制并发
        """
        # 多摄像头模式：并行处理多个摄像头
        if task.is_multi_camera():
            self._download_worker_multi_camera(task)
        else:
            self._download_worker_single_camera(task)

    def _download_worker_single_camera(self, task: TaskInfo):
        """单摄像头下载逻辑"""
        total_segments = len(task.segments)
        max_workers = self.config.max_concurrent_per_camera
        logger.info(f"[并行下载] 开始下载 {total_segments} 个片段 (max_workers={max_workers})")

        self._log(task.task_id, "info",
                  f"启动并行下载: {total_segments}个片段 (同一摄像头最多{max_workers}并发)",
                  category="download")

        completed = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_segment = {
                executor.submit(self._download_segment_return, task, segment): segment
                for segment in task.segments
            }

            for future in as_completed(future_to_segment):
                if self._stop_flag.is_set():
                    logger.info(f"[并行下载] 收到停止信号，取消剩余任务")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                segment = future_to_segment[future]
                try:
                    success = future.result()
                    if success:
                        completed += 1
                        # 下载成功，放入分析队列（传入segment对象，而非元组）
                        self.analyze_queue.put(segment)
                        logger.info(f"[并行下载] 片段#{segment.index} 完成 ({completed}/{total_segments})")
                    else:
                        failed += 1
                        logger.warning(f"[并行下载] 片段#{segment.index} 下载失败")
                except Exception as e:
                    failed += 1
                    logger.error(f"[并行下载] 片段#{segment.index} 异常: {e}")

        logger.info(f"[并行下载] 下载线程结束: 成功{completed}, 失败{failed}")

    def _download_worker_multi_camera(self, task: TaskInfo):
        """多摄像头下载逻辑：摄像头间并行，每个摄像头内部限制并发"""
        camera_count = len(task.camera_tasks)
        total_segments = sum(len(ct.segments) for ct in task.camera_tasks)

        logger.info(f"[多摄像头下载] 开始下载 {camera_count} 个摄像头, 共 {total_segments} 个片段")
        self._log(task.task_id, "info",
                  f"多摄像头并行下载: {camera_count}个摄像头, {total_segments}个片段",
                  category="download")

        # 摄像头间并行
        with ThreadPoolExecutor(max_workers=camera_count) as executor:
            future_to_camera = {
                executor.submit(self._download_camera_segments, task, camera_task): camera_task
                for camera_task in task.camera_tasks
            }

            for future in as_completed(future_to_camera):
                if self._stop_flag.is_set():
                    logger.info(f"[多摄像头下载] 收到停止信号，取消剩余任务")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                camera_task = future_to_camera[future]
                try:
                    future.result()
                    logger.info(f"[多摄像头下载] 摄像头#{camera_task.index} ({camera_task.channel_num}) 完成")
                except Exception as e:
                    logger.error(f"[多摄像头下载] 摄像头#{camera_task.index} 异常: {e}")

        # 汇总统计
        total_completed = sum(ct.completed_count for ct in task.camera_tasks)
        total_failed = sum(ct.failed_count for ct in task.camera_tasks)
        logger.info(f"[多摄像头下载] 下载完成: 成功{total_completed}, 失败{total_failed}")

    def _download_camera_segments(self, task: TaskInfo, camera_task: CameraTaskInfo):
        """下载单个摄像头的所有片段（内部限制并发）"""
        camera_task.status = "running"
        total_segments = len(camera_task.segments)
        max_workers = self.config.max_concurrent_per_camera

        logger.info(f"[摄像头#{camera_task.index}] 开始下载 {total_segments} 个片段 (channel={camera_task.channel_num})")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_segment = {
                executor.submit(self._download_camera_segment_return, task, camera_task, segment): segment
                for segment in camera_task.segments
            }

            for future in as_completed(future_to_segment):
                if self._stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                segment = future_to_segment[future]
                try:
                    success = future.result()
                    if success:
                        camera_task.completed_count += 1
                        # 下载成功，放入分析队列（附加摄像头信息）
                        segment.camera_index = camera_task.index  # 标记来自哪个摄像头
                        segment.camera_channel = camera_task.channel_num
                        self.analyze_queue.put(segment)
                    else:
                        camera_task.failed_count += 1
                except Exception as e:
                    camera_task.failed_count += 1
                    logger.error(f"[摄像头#{camera_task.index}] 片段#{segment.index} 异常: {e}")

        camera_task.status = "completed" if camera_task.failed_count == 0 else "failed"

    def _download_camera_segment_return(self, task: TaskInfo, camera_task: CameraTaskInfo, segment: SegmentInfo) -> bool:
        """下载单个摄像头的单个片段（多摄像头模式）"""
        # 临时修改task的channel_num以使用正确的缓存路径
        original_channel_num = task.channel_num
        task.channel_num = camera_task.channel_num

        try:
            self._download_segment(task, segment)
        finally:
            task.channel_num = original_channel_num

        return segment.download_status == SegmentStatus.COMPLETED

    def _download_segment_return(self, task: TaskInfo, segment: SegmentInfo) -> bool:
        """下载单个片段并返回结果（用于并行下载）"""
        self._download_segment(task, segment)
        return segment.download_status == SegmentStatus.COMPLETED

    # ===== 视频缓存方法 =====

    def _get_device_id_for_cache(self, task: TaskInfo) -> str:
        """
        获取用于缓存路径的设备ID（三模式共用缓存）

        缓存key设计原则：无论使用哪种下载方式，相同的视频片段应该有相同的缓存路径。
        缓存路径格式：{cache_dir}/{road_id}/{device_id}/{date}_{start}_{end}.mp4

        优先级：
        1. 根据 channel_num 反查设备ID（HTTP模式通过通道编号确定设备）
        2. 从 device_mapper 按 device_index 获取（RTSP模式）
        3. 使用 road_id + device_cate + device_index 组合（fallback）
        """
        # 获取可用的 device_mapper
        device_mapper = None
        if self.dual_downloader and self.dual_downloader.device_mapper:
            device_mapper = self.dual_downloader.device_mapper
        elif self._cached_device_mapper:
            device_mapper = self._cached_device_mapper
        elif self.tsingcloud_config and self.tsingcloud_config.device_mapping_file:
            try:
                from .tsingcloud_api import DeviceMapper
                self._cached_device_mapper = DeviceMapper(self.tsingcloud_config.device_mapping_file)
                device_mapper = self._cached_device_mapper
                logger.debug(f"[缓存共享] DeviceMapper 已初始化: {self.tsingcloud_config.device_mapping_file}")
            except Exception as e:
                logger.debug(f"[缓存共享] DeviceMapper 初始化失败: {e}")

        if device_mapper:
            # 方法1：根据 channel_num 反查设备ID（优先，适用于HTTP模式）
            # channel_num 是 HTTP 接口返回的通道编号，如 "11011500581314000972"
            # sn 格式为 "2_xxx_通道编号"，通道编号在末尾
            if task.channel_num:
                try:
                    all_devices = device_mapper.get_all_devices(task.road_id, task.device_cate)
                    for device in all_devices:
                        sn = device.get('sn', '')
                        # sn 格式: 2_11011500002000000001_11011500581314000972
                        # 通道编号是 sn 的最后一部分
                        if sn.endswith(task.channel_num) or task.channel_num in sn:
                            device_id = device.get('deviceId')
                            if device_id:
                                logger.debug(f"[缓存] 根据channel_num={task.channel_num} 匹配到设备 {device_id}")
                                return device_id
                except Exception as e:
                    logger.debug(f"[缓存] 根据channel_num查找设备失败: {e}")

            # 方法2：按 device_index 获取（适用于RTSP模式）
            try:
                device_id = device_mapper.get_device_id(
                    task.road_id, task.device_cate, task.device_index
                )
                if device_id:
                    return device_id
            except Exception:
                pass

        # 方法3：统一的fallback - 使用 road_id + device_cate + device_index
        # 这确保即使没有映射文件，相同参数的请求也会命中相同的缓存
        return f"{task.road_id}_{task.device_cate}_{task.device_index}"

    def _get_cache_path(self, task: TaskInfo, segment: SegmentInfo) -> str:
        """获取视频缓存路径"""
        device_id = self._get_device_id_for_cache(task)
        date_str = segment.start_time.strftime('%Y%m%d')
        start_str = segment.start_time.strftime('%H%M%S')
        end_str = segment.end_time.strftime('%H%M%S')

        cache_dir = os.path.join(
            self.config.video_cache_dir,
            task.road_id,
            device_id
        )
        os.makedirs(cache_dir, exist_ok=True)

        return os.path.join(cache_dir, f"{date_str}_{start_str}_{end_str}.mp4")

    def _check_cache(self, cache_path: str) -> bool:
        """检查缓存是否有效"""
        if not os.path.exists(cache_path):
            return False
        size = os.path.getsize(cache_path)
        return size >= self.config.min_video_size

    def _save_to_cache(self, source_path: str, cache_path: str):
        """将视频保存到缓存"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            shutil.copy2(source_path, cache_path)
            logger.info(f"[缓存写入] 已保存到缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"[缓存写入] 保存失败: {e}")

    # ===== 下载方法 =====

    def _download_segment(self, task: TaskInfo, segment: SegmentInfo):
        """下载单个片段 - 使用双账号下载器（先RTSP后HTTP轮询）"""

        # ===== 1. 检查缓存 =====
        if self.config.enable_video_cache:
            cache_path = self._get_cache_path(task, segment)
            if self._check_cache(cache_path):
                file_size = os.path.getsize(cache_path) / (1024 * 1024)
                logger.info(f"[缓存命中] 片段#{segment.index} 复用缓存 ({file_size:.1f}MB): {cache_path}")
                self._log(task.task_id, "success",
                          f"片段#{segment.index} 缓存命中 ({file_size:.1f}MB)",
                          segment.index, {"cache_hit": True}, category="download")

                segment.video_path = cache_path
                segment.download_status = SegmentStatus.COMPLETED

                self._emit_event(EventType.PROGRESS, {
                    "task_id": task.task_id,
                    "type": "download",
                    "segment": segment.index,
                    "total": len(task.segments),
                    "time_range": segment.time_range,
                    "status": "completed",
                    "message": f"片段 #{segment.index} 缓存命中"
                })
                return

        # ===== 2. 开始下载 =====
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

        # 视频输出路径
        video_filename = f"segment_{segment.index:03d}_{segment.time_range.replace(':', '').replace('-', '_').replace('/', '')}.mp4"
        video_path = os.path.join(self.config.temp_dir, task.task_id, video_filename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        retry_count = 0
        max_retries = self.config.download_retry_count

        # ===== 根据下载方式选择策略 =====
        # download_method: "auto"(先RTSP后HTTP), "rtsp"(仅RTSP), "http"(仅HTTP轮询)
        use_dual_downloader = self.dual_downloader and task.download_method in ("auto", "rtsp")

        if use_dual_downloader:
            try:
                method_desc = {
                    "auto": "双账号模式（先RTSP后HTTP）",
                    "rtsp": "RTSP模式"
                }.get(task.download_method, "双账号模式")

                self._log(task.task_id, "info",
                          f"片段#{segment.index} 使用{method_desc}...",
                          segment.index, category="download")

                # 进度回调
                def dual_progress(method: str, downloaded: int, total: int):
                    if method == "rtsp":
                        if downloaded == 0 and total == 0:
                            self._log(task.task_id, "info",
                                      f"片段#{segment.index} 正在RTSP录制...",
                                      segment.index, category="download")
                        elif downloaded > 0:
                            self._log(task.task_id, "info",
                                      f"片段#{segment.index} RTSP录制: {downloaded/(1024*1024):.1f}MB",
                                      segment.index, category="download")
                    elif method == "http_poll":
                        self._log(task.task_id, "info",
                                  f"片段#{segment.index} HTTP轮询 {downloaded}/{total}...",
                                  segment.index, category="download")
                    elif method == "http":
                        if total > 0:
                            self._log(task.task_id, "info",
                                      f"片段#{segment.index} HTTP下载: {downloaded/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB",
                                      segment.index, category="download")

                # 根据下载方式决定是否优先RTSP
                prefer_rtsp = task.download_method in ("auto", "rtsp")

                success = self.dual_downloader.download_video(
                    road_id=task.road_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    output_path=video_path,
                    prefer_rtsp=prefer_rtsp,
                    device_cate=task.device_cate,
                    device_index=task.device_index,
                    channel_num=task.channel_num,
                    progress_callback=dual_progress
                )

                if success and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    segment.video_path = video_path
                    segment.download_status = SegmentStatus.COMPLETED

                    file_size = os.path.getsize(video_path) / (1024 * 1024)
                    self._log(task.task_id, "success",
                              f"片段#{segment.index} 下载完成 ({file_size:.1f}MB)",
                              segment.index, {"file_size_mb": file_size, "method": "dual"}, category="download")

                    # 保存到缓存
                    if self.config.enable_video_cache:
                        cache_path = self._get_cache_path(task, segment)
                        self._save_to_cache(video_path, cache_path)

                    # 注意：不在这里入队，由 _download_worker 统一处理入队
                    self._update_queue_status(task)
                    return

                # 双账号下载失败
                if task.download_method == "rtsp":
                    # 仅RTSP模式：不回退，直接标记失败
                    self._log(task.task_id, "error",
                              f"片段#{segment.index} RTSP下载失败（仅RTSP模式，不回退HTTP）",
                              segment.index, category="download")
                    segment.download_status = SegmentStatus.DOWNLOAD_FAILED
                    segment.error_message = "RTSP下载失败"
                    self._emit_event(EventType.ERROR, {
                        "task_id": task.task_id,
                        "segment": segment.index,
                        "type": "download",
                        "message": "RTSP下载失败（仅RTSP模式）",
                        "will_retry": False
                    })
                    return
                else:
                    # auto模式：回退到HTTP轮询
                    self._log(task.task_id, "warning",
                              f"片段#{segment.index} 双账号下载失败，尝试传统HTTP轮询...",
                              segment.index, category="download")

            except Exception as e:
                if task.download_method == "rtsp":
                    # 仅RTSP模式：不回退
                    self._log(task.task_id, "error",
                              f"片段#{segment.index} RTSP异常: {e}（仅RTSP模式，不回退）",
                              segment.index, category="download")
                    segment.download_status = SegmentStatus.DOWNLOAD_FAILED
                    segment.error_message = str(e)
                    return
                else:
                    self._log(task.task_id, "warning",
                              f"片段#{segment.index} 双账号下载异常: {e}，回退到传统HTTP轮询...",
                              segment.index, category="download")

        # ===== 回退: 传统HTTP轮询下载（仅auto和http模式）=====
        if task.download_method == "rtsp":
            # 仅RTSP模式不应该到这里，保护性返回
            return
        while retry_count <= max_retries:
            try:
                self._log(task.task_id, "info",
                          f"片段#{segment.index} 使用HTTP轮询方式下载...",
                          segment.index, category="download")

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
                              f"片段#{segment.index} HTTP下载完成 ({file_size:.1f}MB)",
                              segment.index, {"file_size_mb": file_size, "method": "http"}, category="download")

                    # 保存到缓存
                    if self.config.enable_video_cache:
                        cache_path = self._get_cache_path(task, segment)
                        self._save_to_cache(video_path, cache_path)

                    # 注意：不在这里入队，由 _download_worker 统一处理入队
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
        """清理无事件的片段（不删除缓存文件）"""
        if segment.video_path and os.path.exists(segment.video_path):
            # 检查是否为缓存文件，如果是则不删除
            if self.config.enable_video_cache:
                cache_dir = os.path.normpath(self.config.video_cache_dir)
                video_dir = os.path.normpath(os.path.dirname(segment.video_path))
                if video_dir.startswith(cache_dir):
                    logger.debug(f"[清理] 跳过缓存文件: {segment.video_path}")
                    return

            try:
                os.remove(segment.video_path)
                segment.video_path = None
                logger.debug(f"[清理] 已删除临时文件: {segment.video_path}")
            except Exception as e:
                logger.warning(f"清理视频失败: {e}")

    def _update_queue_status(self, task: TaskInfo, force: bool = False):
        """更新并发送队列状态（支持多摄像头模式，带限流）

        Args:
            task: 任务信息
            force: 强制发送（跳过限流）
        """
        # 限流：最多每0.5秒发送一次（除非force=True）
        current_time = time.time()
        if not force:
            last_update = getattr(task, '_last_queue_update', 0)
            if current_time - last_update < 0.5:
                return
        task._last_queue_update = current_time

        if task.is_multi_camera():
            # 多摄像头模式：发送摘要而非完整片段列表（优化大任务）
            total_segments = sum(len(ct.segments) for ct in task.camera_tasks)

            completed = 0
            for ct in task.camera_tasks:
                completed += sum(1 for s in ct.segments
                                if s.download_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED]
                                and s.analyze_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED])

            # 如果片段数超过100，只发送摘要
            if total_segments > 100:
                self._emit_event(EventType.QUEUE, {
                    "task_id": task.task_id,
                    "multi_camera": True,
                    "camera_count": len(task.camera_tasks),
                    "camera_tasks": [ct.to_dict() for ct in task.camera_tasks],
                    "segments": [],  # 不发送完整列表
                    "completed": completed,
                    "total": total_segments,
                    "summary_mode": True  # 标记为摘要模式
                })
            else:
                all_segments = []
                for ct in task.camera_tasks:
                    all_segments.extend(ct.segments)
                self._emit_event(EventType.QUEUE, {
                    "task_id": task.task_id,
                    "multi_camera": True,
                    "camera_count": len(task.camera_tasks),
                    "camera_tasks": [ct.to_dict() for ct in task.camera_tasks],
                    "segments": [s.to_dict() for s in all_segments],
                    "completed": completed,
                    "total": total_segments
                })
        else:
            # 单摄像头模式
            completed = sum(1 for s in task.segments
                           if s.download_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED]
                           and s.analyze_status in [SegmentStatus.COMPLETED, SegmentStatus.SKIPPED])

            # 如果片段数超过100，只发送摘要
            if len(task.segments) > 100:
                self._emit_event(EventType.QUEUE, {
                    "task_id": task.task_id,
                    "segments": [],
                    "completed": completed,
                    "total": len(task.segments),
                    "summary_mode": True
                })
            else:
                self._emit_event(EventType.QUEUE, {
                    "task_id": task.task_id,
                    "segments": [s.to_dict() for s in task.segments],
                    "completed": completed,
                    "total": len(task.segments)
                })

    def _generate_report(self, task: TaskInfo):
        """生成HTML报告（支持多摄像头模式）"""
        task_dir = os.path.join(self.config.result_dir, task.task_id)
        report_path = os.path.join(task_dir, "report.html")

        # 收集事件（兼容多摄像头模式）
        events = []
        if task.is_multi_camera():
            all_segments = []
            for ct in task.camera_tasks:
                all_segments.extend(ct.segments)
            total_segments = len(all_segments)
            camera_info = f"多摄像头模式（{len(task.camera_tasks)}个）"
        else:
            all_segments = task.segments
            total_segments = len(task.segments)
            camera_info = task.channel_num

        for seg in all_segments:
            if seg.result == "detected":
                camera_label = f" [摄像头{seg.camera_index}]" if seg.camera_index is not None else ""
                events.append({
                    "index": seg.index,
                    "time_range": seg.time_range + camera_label,
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
            <p><strong>路口:</strong> {task.road_id} | <strong>摄像头:</strong> {camera_info}</p>
            <p><strong>时间段:</strong> {task.start_date} {task.start_time} → {task.end_date} {task.end_time}</p>
            <p><strong>分析模式:</strong> {"交通事故检测" if task.mode == "accident" else "交通违法检测"}</p>
            <p><strong>VLM模型:</strong> {task.model}</p>
            <hr>
            <p><strong>总片段数:</strong> {total_segments}</p>
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
        """获取任务状态（支持多摄像头模式）"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        if task.is_multi_camera():
            all_segments = []
            for ct in task.camera_tasks:
                all_segments.extend(ct.segments)
            completed = sum(1 for s in all_segments
                           if s.analyze_status == SegmentStatus.COMPLETED)
            return {
                **task.to_dict(),
                "completed_segments": completed,
                "segments": [s.to_dict() for s in all_segments]
            }
        else:
            completed = sum(1 for s in task.segments
                           if s.analyze_status == SegmentStatus.COMPLETED)
            return {
                **task.to_dict(),
                "completed_segments": completed,
                "segments": [s.to_dict() for s in task.segments]
            }

    def stream_progress(self, task_id: str) -> Generator[str, None, None]:
        """SSE进度流生成器（支持多摄像头模式）"""
        task = self.tasks.get(task_id)
        if not task:
            yield f"event: error\ndata: {{\"message\": \"任务不存在\"}}\n\n"
            return

        # 初始状态（兼容多摄像头模式）
        if task.is_multi_camera():
            all_segments = []
            for ct in task.camera_tasks:
                all_segments.extend(ct.segments)
            yield f"event: queue\ndata: {json.dumps({'segments': [s.to_dict() for s in all_segments], 'completed': 0, 'total': len(all_segments), 'multi_camera': True, 'camera_count': len(task.camera_tasks)})}\n\n"
        else:
            yield f"event: queue\ndata: {json.dumps({'segments': [s.to_dict() for s in task.segments], 'completed': 0, 'total': len(task.segments)})}\n\n"

        # 持续推送直到任务完成
        while task.status == "running":
            time.sleep(1)
            # 实际事件通过event_callback推送

        # 最终状态
        yield f"event: complete\ndata: {json.dumps(task.to_dict())}\n\n"
