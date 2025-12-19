"""
批量视频遍历处理器

支持两种遍历模式：
1. 时间遍历模式：遍历所有路口的全景摄像头
2. 路口遍历模式：遍历指定路口的全景摄像头

核心特性：
- 容错机制：路口/摄像头不可用时跳过继续
- 层级进度：总进度 → 路口进度 → 摄像头进度 → 片段进度
- SSE事件推送
"""
from __future__ import annotations

import os
import csv
import json
import uuid
import logging
import threading
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .tsingcloud_api import TsingcloudAPI, CameraInfo, TsingcloudAPIError
from .config import BatchProcessConfig, HistoryProcessConfig
from .history_video_processor import HistoryVideoProcessor, TaskInfo, SegmentInfo, SegmentStatus, EventType

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """批量任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class RoadStatus(Enum):
    """路口状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class CameraStatus(Enum):
    """摄像头状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class BatchEventType(Enum):
    """批量任务SSE事件类型"""
    BATCH_START = "batch_start"
    BATCH_PROGRESS = "batch_progress"
    BATCH_COMPLETE = "batch_complete"
    BATCH_ERROR = "batch_error"
    ROAD_START = "road_start"
    ROAD_PROGRESS = "road_progress"
    ROAD_COMPLETE = "road_complete"
    ROAD_SKIPPED = "road_skipped"
    CAMERA_START = "camera_start"
    CAMERA_COMPLETE = "camera_complete"
    CAMERA_SKIPPED = "camera_skipped"
    LOG = "log"
    RESULT = "result"


@dataclass
class CameraTask:
    """摄像头级别任务"""
    camera_info: CameraInfo
    status: CameraStatus = CameraStatus.PENDING
    events_found: int = 0
    events_cleared: int = 0
    total_segments: int = 0
    completed_segments: int = 0
    error_message: Optional[str] = None
    task_id: Optional[str] = None  # 关联的 HistoryVideoProcessor 任务ID

    def to_dict(self) -> dict:
        return {
            "channel_num": self.camera_info.channel_num,
            "camera_type": self.camera_info.camera_type,
            "camera_type_str": self.camera_info.camera_type_str,
            "is_panoramic": self.camera_info.is_panoramic,
            "status": self.status.value,
            "events_found": self.events_found,
            "events_cleared": self.events_cleared,
            "total_segments": self.total_segments,
            "completed_segments": self.completed_segments,
            "error": self.error_message
        }


@dataclass
class RoadTask:
    """路口级别任务"""
    road_id: str
    road_name: str
    cameras: List[CameraTask] = field(default_factory=list)
    status: RoadStatus = RoadStatus.PENDING
    total_cameras: int = 0
    completed_cameras: int = 0
    skipped_cameras: int = 0
    events_found: int = 0
    events_cleared: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "road_id": self.road_id,
            "road_name": self.road_name,
            "status": self.status.value,
            "total_cameras": self.total_cameras,
            "completed_cameras": self.completed_cameras,
            "skipped_cameras": self.skipped_cameras,
            "events_found": self.events_found,
            "events_cleared": self.events_cleared,
            "error": self.error_message,
            "cameras": [c.to_dict() for c in self.cameras]
        }


@dataclass
class BatchTaskInfo:
    """批量任务信息（支持跨日期时间段）"""
    batch_id: str
    mode: str  # "time_traverse" | "road_traverse"
    start_date: str  # 开始日期，如 "2024-12-17"
    start_time: str  # 开始时间，如 "20:00"
    end_date: str    # 结束日期，如 "2024-12-19"
    end_time: str    # 结束时间，如 "08:00"
    road_ids: List[str]  # 空列表=所有路口
    model: str
    analysis_mode: str  # "accident" | "violation"
    violation_types: List[str] = field(default_factory=list)
    segment_duration: int = 300

    roads: List[RoadTask] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING

    # 统计
    total_roads: int = 0
    completed_roads: int = 0
    skipped_roads: int = 0
    total_cameras: int = 0
    completed_cameras: int = 0
    skipped_cameras: int = 0
    total_events_found: int = 0
    total_events_cleared: int = 0

    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "mode": self.mode,
            "start_date": self.start_date,
            "start_time": self.start_time,
            "end_date": self.end_date,
            "end_time": self.end_time,
            "model": self.model,
            "analysis_mode": self.analysis_mode,
            "status": self.status.value,
            "total_roads": self.total_roads,
            "completed_roads": self.completed_roads,
            "skipped_roads": self.skipped_roads,
            "total_cameras": self.total_cameras,
            "completed_cameras": self.completed_cameras,
            "skipped_cameras": self.skipped_cameras,
            "total_events_found": self.total_events_found,
            "total_events_cleared": self.total_events_cleared,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None
        }

    def to_dict_with_roads(self) -> dict:
        result = self.to_dict()
        result["roads"] = [r.to_dict() for r in self.roads]
        return result


class BatchVideoProcessor:
    """
    批量视频处理器

    管理多路口多摄像头的批量分析任务：
    - 支持时间遍历和路口遍历两种模式
    - 容错机制：路口/摄像头失败时跳过
    - 复用 HistoryVideoProcessor 处理单摄像头
    """

    def __init__(
        self,
        api: TsingcloudAPI,
        batch_config: BatchProcessConfig,
        history_config: HistoryProcessConfig,
        pipeline_func: Callable = None,
        event_callback: Callable[[BatchEventType, dict], None] = None,
        roads_csv_path: str = None
    ):
        """
        初始化批量处理器

        Args:
            api: 云控智行API客户端
            batch_config: 批量处理配置
            history_config: 历史视频处理配置
            pipeline_func: 视频分析函数
            event_callback: SSE事件回调
            roads_csv_path: 路口CSV文件路径
        """
        self.api = api
        self.batch_config = batch_config
        self.history_config = history_config
        self.pipeline_func = pipeline_func
        self.event_callback = event_callback

        # 路口CSV路径
        if roads_csv_path:
            self.roads_csv_path = roads_csv_path
        else:
            self.roads_csv_path = os.path.join(
                os.path.dirname(__file__),
                '..', '车网路口视频流相关资料', 'rcuid.csv'
            )

        # 任务管理
        self.tasks: Dict[str, BatchTaskInfo] = {}
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()

        # 确保目录存在
        batch_config.ensure_dirs()

    def _emit_event(self, event_type: BatchEventType, data: dict):
        """发送SSE事件"""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"事件回调失败: {e}")

    def _log(self, batch_id: str, level: str, message: str, category: str = "general"):
        """记录日志并发送事件"""
        logger.info(f"[Batch {batch_id}] {message}")
        self._emit_event(BatchEventType.LOG, {
            "batch_id": batch_id,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
            "category": category
        })

    def _load_all_roads(self) -> List[Dict[str, str]]:
        """从CSV加载所有路口"""
        roads = {}

        try:
            with open(self.roads_csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    road_id = row.get('id', '').strip()
                    if road_id and road_id not in roads:
                        roads[road_id] = {
                            'road_id': road_id,
                            'rcu_id': row.get('rcuId', ''),
                            'road_name': f'路口 #{road_id}'
                        }
        except Exception as e:
            logger.error(f"加载路口CSV失败: {e}")
            return []

        # 按ID排序
        return sorted(
            roads.values(),
            key=lambda x: int(x['road_id']) if x['road_id'].isdigit() else 0
        )

    def _get_panoramic_cameras(
        self,
        road_id: str,
        start_str: str,
        end_str: str
    ) -> List[CameraInfo]:
        """获取路口的全景摄像头列表"""
        try:
            cameras = self.api.get_road_cameras(road_id, start_str, end_str)

            if self.batch_config.camera_filter == "panoramic":
                return [c for c in cameras if c.is_panoramic]
            return cameras

        except TsingcloudAPIError as e:
            logger.warning(f"路口 {road_id} 获取摄像头失败: {e}")
            return []
        except Exception as e:
            logger.error(f"路口 {road_id} 获取摄像头异常: {e}")
            return []

    def create_batch_task(
        self,
        mode: str,
        start_date: str,
        start_time: str,
        end_date: str,
        end_time: str,
        road_ids: List[str] = None,
        model: str = "qwen-vl-plus",
        analysis_mode: str = "accident",
        violation_types: List[str] = None,
        segment_duration: int = None
    ) -> BatchTaskInfo:
        """
        创建批量任务（支持跨日期时间段）

        Args:
            mode: "time_traverse" (遍历所有路口) 或 "road_traverse" (遍历指定路口)
            start_date: 开始日期 "2024-12-17"
            start_time: 开始时间 "20:00"
            end_date: 结束日期 "2024-12-19"
            end_time: 结束时间 "08:00"
            road_ids: 路口ID列表，为空则遍历所有
            model: VLM模型
            analysis_mode: "accident" 或 "violation"
            violation_types: 违法类型列表
            segment_duration: 分片时长（秒）

        Returns:
            BatchTaskInfo
        """
        batch_id = str(uuid.uuid4())[:8]
        segment_duration = segment_duration or self.history_config.segment_duration

        # 确定路口列表
        if mode == "time_traverse" or not road_ids:
            # 时间遍历模式：加载所有路口
            all_roads = self._load_all_roads()
            road_list = [r['road_id'] for r in all_roads]
            road_names = {r['road_id']: r['road_name'] for r in all_roads}
        else:
            # 路口遍历模式：使用指定路口
            road_list = road_ids
            road_names = {rid: f'路口 #{rid}' for rid in road_ids}

        # 限制路口数量
        if len(road_list) > self.batch_config.max_roads_per_batch:
            road_list = road_list[:self.batch_config.max_roads_per_batch]
            logger.warning(f"路口数量超限，截取前 {self.batch_config.max_roads_per_batch} 个")

        # 创建路口任务列表
        road_tasks = []
        for rid in road_list:
            road_tasks.append(RoadTask(
                road_id=rid,
                road_name=road_names.get(rid, f'路口 #{rid}')
            ))

        # 创建批量任务（支持跨日期）
        task = BatchTaskInfo(
            batch_id=batch_id,
            mode=mode,
            start_date=start_date,
            start_time=start_time,
            end_date=end_date,
            end_time=end_time,
            road_ids=road_ids or [],
            model=model,
            analysis_mode=analysis_mode,
            violation_types=violation_types or [],
            segment_duration=segment_duration,
            roads=road_tasks,
            total_roads=len(road_tasks)
        )

        self.tasks[batch_id] = task
        return task

    def start_batch_task(self, batch_id: str):
        """启动批量任务"""
        task = self.tasks.get(batch_id)
        if not task:
            raise ValueError(f"批量任务不存在: {batch_id}")

        # 重置停止标志
        self._stop_flag.clear()

        task.status = BatchStatus.RUNNING
        task.started_at = datetime.now()

        # 计算时间跨度描述
        if task.start_date == task.end_date:
            time_range_desc = f"{task.start_date} {task.start_time}-{task.end_time}"
        else:
            time_range_desc = f"{task.start_date} {task.start_time} → {task.end_date} {task.end_time}"

        self._log(batch_id, "info",
                  f"批量任务启动 - 模式:{task.mode}, 路口数:{task.total_roads}, "
                  f"时间段:{time_range_desc}")

        # 发送开始事件
        self._emit_event(BatchEventType.BATCH_START, {
            "batch_id": batch_id,
            "mode": task.mode,
            "total_roads": task.total_roads,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "time_range": time_range_desc
        })

        # 构建时间字符串（用于API调用：使用开始日期作为基准）
        # 注：对于跨日期范围，下载器会自动处理分片
        start_str = f"{task.start_date.replace('-', '')}{task.start_time.replace(':', '')}00"
        end_str = f"{task.end_date.replace('-', '')}{task.end_time.replace(':', '')}00"

        # 遍历路口
        for road_task in task.roads:
            if self._stop_flag.is_set():
                self._log(batch_id, "warning", "批量任务被停止")
                break

            self._process_road(task, road_task, start_str, end_str)

        # 任务完成
        task.finished_at = datetime.now()
        if self._stop_flag.is_set():
            task.status = BatchStatus.STOPPED
        else:
            task.status = BatchStatus.COMPLETED

        # 生成报告
        if self.batch_config.generate_summary_report:
            report_path = self._generate_batch_report(task)
        else:
            report_path = None

        # 发送完成事件
        self._emit_event(BatchEventType.BATCH_COMPLETE, {
            "batch_id": batch_id,
            "status": task.status.value,
            "total_roads": task.total_roads,
            "completed_roads": task.completed_roads,
            "skipped_roads": task.skipped_roads,
            "total_events_found": task.total_events_found,
            "total_events_cleared": task.total_events_cleared,
            "report_url": f"/api/batch/report/{batch_id}" if report_path else None,
            "duration_seconds": (task.finished_at - task.started_at).total_seconds()
        })

        self._log(batch_id, "info",
                  f"批量任务完成 - 完成:{task.completed_roads}, 跳过:{task.skipped_roads}, "
                  f"检出事件:{task.total_events_found}")

    def _process_road(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        start_str: str,
        end_str: str
    ):
        """处理单个路口"""
        batch_id = batch_task.batch_id
        road_id = road_task.road_id

        road_task.status = RoadStatus.RUNNING

        self._log(batch_id, "info", f"开始处理路口 {road_id}", category="road")

        # 发送路口开始事件
        self._emit_event(BatchEventType.ROAD_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "road_name": road_task.road_name,
            "road_index": batch_task.roads.index(road_task),
            "total_roads": batch_task.total_roads
        })

        try:
            # 获取全景摄像头列表
            cameras = self._get_panoramic_cameras(road_id, start_str, end_str)

            if not cameras:
                # 无全景摄像头，跳过
                road_task.status = RoadStatus.SKIPPED
                road_task.error_message = "无可用全景摄像头"
                batch_task.skipped_roads += 1

                self._log(batch_id, "warning",
                          f"路口 {road_id} 无全景摄像头，跳过", category="road")

                self._emit_event(BatchEventType.ROAD_SKIPPED, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "reason": "无可用全景摄像头"
                })
                return

            # 限制摄像头数量
            if len(cameras) > self.batch_config.max_cameras_per_road:
                cameras = cameras[:self.batch_config.max_cameras_per_road]

            road_task.total_cameras = len(cameras)

            # 创建摄像头任务
            for cam in cameras:
                road_task.cameras.append(CameraTask(camera_info=cam))

            # 遍历摄像头
            for camera_task in road_task.cameras:
                if self._stop_flag.is_set():
                    break

                try:
                    self._process_camera(batch_task, road_task, camera_task)
                except Exception as e:
                    # 单个摄像头失败，记录错误，继续下一个
                    camera_task.status = CameraStatus.FAILED
                    camera_task.error_message = str(e)
                    road_task.skipped_cameras += 1
                    batch_task.skipped_cameras += 1

                    self._log(batch_id, "error",
                              f"摄像头 {camera_task.camera_info.channel_num} 处理失败: {e}",
                              category="camera")

                    self._emit_event(BatchEventType.CAMERA_SKIPPED, {
                        "batch_id": batch_id,
                        "road_id": road_id,
                        "channel_num": camera_task.camera_info.channel_num,
                        "reason": str(e)
                    })

            # 路口完成
            road_task.status = RoadStatus.COMPLETED
            batch_task.completed_roads += 1

            self._log(batch_id, "success",
                      f"路口 {road_id} 处理完成 - 检出:{road_task.events_found}",
                      category="road")

            self._emit_event(BatchEventType.ROAD_COMPLETE, {
                "batch_id": batch_id,
                "road_id": road_id,
                "total_cameras": road_task.total_cameras,
                "completed_cameras": road_task.completed_cameras,
                "skipped_cameras": road_task.skipped_cameras,
                "events_found": road_task.events_found
            })

        except TsingcloudAPIError as e:
            # API错误，跳过该路口
            road_task.status = RoadStatus.SKIPPED
            road_task.error_message = str(e)
            batch_task.skipped_roads += 1

            self._log(batch_id, "error",
                      f"路口 {road_id} API错误: {e}", category="road")

            self._emit_event(BatchEventType.ROAD_SKIPPED, {
                "batch_id": batch_id,
                "road_id": road_id,
                "reason": str(e)
            })

        except Exception as e:
            # 其他错误
            road_task.status = RoadStatus.FAILED
            road_task.error_message = str(e)
            batch_task.skipped_roads += 1

            self._log(batch_id, "error",
                      f"路口 {road_id} 处理异常: {e}", category="road")

            self._emit_event(BatchEventType.ROAD_SKIPPED, {
                "batch_id": batch_id,
                "road_id": road_id,
                "reason": str(e)
            })

        # 更新批量进度
        self._emit_event(BatchEventType.BATCH_PROGRESS, {
            "batch_id": batch_id,
            "completed_roads": batch_task.completed_roads,
            "skipped_roads": batch_task.skipped_roads,
            "total_roads": batch_task.total_roads,
            "total_events_found": batch_task.total_events_found
        })

    def _process_camera(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        camera_task: CameraTask
    ):
        """处理单个摄像头"""
        batch_id = batch_task.batch_id
        road_id = road_task.road_id
        channel_num = camera_task.camera_info.channel_num

        camera_task.status = CameraStatus.RUNNING

        self._log(batch_id, "info",
                  f"开始处理摄像头 {channel_num} (路口 {road_id})",
                  category="camera")

        # 发送摄像头开始事件
        self._emit_event(BatchEventType.CAMERA_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "camera_index": road_task.cameras.index(camera_task),
            "total_cameras": road_task.total_cameras
        })

        # 创建事件收集器
        events_found = 0
        events_cleared = 0

        def camera_event_callback(event_type: EventType, data: dict):
            """摄像头事件回调，转发到批量事件"""
            nonlocal events_found, events_cleared

            # 统计结果
            if event_type == EventType.RESULT:
                events_found += 1
                # 转发结果事件
                self._emit_event(BatchEventType.RESULT, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "channel_num": channel_num,
                    **data
                })

            elif event_type == EventType.QUEUE:
                # 更新片段统计
                camera_task.total_segments = data.get("total", 0)
                camera_task.completed_segments = data.get("completed", 0)

            elif event_type == EventType.LOG:
                # 转发日志（添加路口和摄像头信息）
                self._emit_event(BatchEventType.LOG, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "channel_num": channel_num,
                    **data
                })

            elif event_type == EventType.COMPLETE:
                events_cleared = data.get("events_cleared", 0)

        # 创建 HistoryVideoProcessor 处理单摄像头
        processor = HistoryVideoProcessor(
            api=self.api,
            config=self.history_config,
            pipeline_func=self.pipeline_func,
            event_callback=camera_event_callback
        )

        # 创建任务（支持跨日期时间段）
        task = processor.create_task(
            road_id=road_id,
            channel_num=channel_num,
            start_date=batch_task.start_date,
            start_time=batch_task.start_time,
            end_date=batch_task.end_date,
            end_time=batch_task.end_time,
            mode=batch_task.analysis_mode,
            model=batch_task.model,
            violation_types=batch_task.violation_types,
            segment_duration=batch_task.segment_duration
        )

        camera_task.task_id = task.task_id
        camera_task.total_segments = len(task.segments)

        # 启动任务（阻塞直到完成）
        processor.start_task(task.task_id)

        # 更新统计
        camera_task.events_found = events_found
        camera_task.events_cleared = events_cleared
        camera_task.status = CameraStatus.COMPLETED

        road_task.events_found += events_found
        road_task.events_cleared += events_cleared
        road_task.completed_cameras += 1

        batch_task.total_events_found += events_found
        batch_task.total_events_cleared += events_cleared
        batch_task.completed_cameras += 1

        self._log(batch_id, "success",
                  f"摄像头 {channel_num} 处理完成 - 检出:{events_found}",
                  category="camera")

        # 发送摄像头完成事件
        self._emit_event(BatchEventType.CAMERA_COMPLETE, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "events_found": events_found,
            "events_cleared": events_cleared,
            "total_segments": camera_task.total_segments
        })

    def stop_batch_task(self, batch_id: str):
        """停止批量任务"""
        self._stop_flag.set()
        task = self.tasks.get(batch_id)
        if task:
            task.status = BatchStatus.STOPPED
            self._log(batch_id, "warning", "批量任务已停止")

    def skip_road(self, batch_id: str, road_id: str) -> bool:
        """跳过指定路口"""
        task = self.tasks.get(batch_id)
        if not task:
            return False

        for road in task.roads:
            if road.road_id == road_id and road.status == RoadStatus.PENDING:
                road.status = RoadStatus.SKIPPED
                road.error_message = "用户跳过"
                task.skipped_roads += 1
                return True

        return False

    def get_batch_status(self, batch_id: str) -> Optional[dict]:
        """获取批量任务状态"""
        task = self.tasks.get(batch_id)
        if not task:
            return None
        return task.to_dict_with_roads()

    def _generate_batch_report(self, task: BatchTaskInfo) -> str:
        """生成批量任务汇总报告"""
        report_dir = os.path.join(self.batch_config.batch_result_dir, task.batch_id)
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "report.html")

        # 收集所有检出事件
        events = []
        for road in task.roads:
            for camera in road.cameras:
                if camera.events_found > 0:
                    events.append({
                        "road_id": road.road_id,
                        "road_name": road.road_name,
                        "channel_num": camera.camera_info.channel_num,
                        "events_found": camera.events_found
                    })

        # 计算耗时
        duration = ""
        if task.started_at and task.finished_at:
            delta = task.finished_at - task.started_at
            hours = int(delta.total_seconds() // 3600)
            minutes = int((delta.total_seconds() % 3600) // 60)
            seconds = int(delta.total_seconds() % 60)
            duration = f"{hours}小时{minutes}分{seconds}秒"

        # 生成HTML报告
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>批量分析报告 - {task.batch_id}</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stat-card .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
        .stat-card .label {{ color: #666; font-size: 14px; margin-top: 5px; }}
        .road-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .road-table th, .road-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        .road-table th {{ background: #f8f9fa; }}
        .status-completed {{ color: #28a745; }}
        .status-skipped {{ color: #ffc107; }}
        .status-failed {{ color: #dc3545; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 4px; font-size: 12px; }}
        .badge-warning {{ background: #fff3e0; color: #e65100; }}
        .badge-success {{ background: #e8f5e9; color: #2e7d32; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>批量分析报告</h1>

        <div class="summary">
            <div class="stat-card">
                <div class="value">{task.total_roads}</div>
                <div class="label">总路口数</div>
            </div>
            <div class="stat-card">
                <div class="value">{task.completed_roads}</div>
                <div class="label">已完成</div>
            </div>
            <div class="stat-card">
                <div class="value">{task.skipped_roads}</div>
                <div class="label">已跳过</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: #e65100;">{task.total_events_found}</div>
                <div class="label">检出事件</div>
            </div>
        </div>

        <h2>任务信息</h2>
        <table class="road-table">
            <tr><th>批量任务ID</th><td>{task.batch_id}</td></tr>
            <tr><th>遍历模式</th><td>{"时间遍历（全部路口）" if task.mode == "time_traverse" else "路口遍历"}</td></tr>
            <tr><th>时间范围</th><td>{task.start_date} {task.start_time} → {task.end_date} {task.end_time}</td></tr>
            <tr><th>分析模式</th><td>{"交通事故检测" if task.analysis_mode == "accident" else "交通违法检测"}</td></tr>
            <tr><th>VLM模型</th><td>{task.model}</td></tr>
            <tr><th>总耗时</th><td>{duration}</td></tr>
        </table>

        <h2>路口处理详情</h2>
        <table class="road-table">
            <thead>
                <tr>
                    <th>路口ID</th>
                    <th>路口名称</th>
                    <th>摄像头数</th>
                    <th>状态</th>
                    <th>检出事件</th>
                    <th>备注</th>
                </tr>
            </thead>
            <tbody>
                {"".join(f'''
                <tr>
                    <td>{road.road_id}</td>
                    <td>{road.road_name}</td>
                    <td>{road.total_cameras}</td>
                    <td class="status-{road.status.value}">{road.status.value}</td>
                    <td><span class="badge badge-{"warning" if road.events_found > 0 else "success"}">{road.events_found}</span></td>
                    <td>{road.error_message or "-"}</td>
                </tr>
                ''' for road in task.roads)}
            </tbody>
        </table>

        <h2>检出事件汇总</h2>
        {f'''
        <table class="road-table">
            <thead>
                <tr><th>路口</th><th>摄像头</th><th>检出数</th></tr>
            </thead>
            <tbody>
                {"".join(f'<tr><td>{e["road_name"]}</td><td>{e["channel_num"]}</td><td>{e["events_found"]}</td></tr>' for e in events)}
            </tbody>
        </table>
        ''' if events else '<p>未检出任何事件</p>'}

        <p style="color: #999; text-align: center; margin-top: 30px;">
            生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
    </div>
</body>
</html>"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 同时保存JSON数据
        json_path = os.path.join(report_dir, "data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(task.to_dict_with_roads(), f, ensure_ascii=False, indent=2)

        self._log(task.batch_id, "info", f"报告已生成: {report_path}")
        return report_path
