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
import random
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
class AccidentTimeSlot:
    """事故时段信息"""
    start_time: str  # 事故检出的开始时间 "HH:MM"
    end_time: str    # 事故检出的结束时间 "HH:MM"
    segment_index: int = 0  # 分片索引
    confidence: float = 0.0  # 置信度
    camera_channel: str = ""  # 检出此事故的摄像头


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
    # P0优化：记录检出事故的时段
    accident_time_slots: List[AccidentTimeSlot] = field(default_factory=list)

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
            "cameras": [c.to_dict() for c in self.cameras],
            "accident_time_slots": [
                {
                    "start_time": slot.start_time,
                    "end_time": slot.end_time,
                    "segment_index": slot.segment_index,
                    "confidence": slot.confidence,
                    "camera_channel": slot.camera_channel
                }
                for slot in self.accident_time_slots
            ]
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

    # 高峰时段优先遍历（前端传入）
    peak_hours_enabled: bool = False  # 是否启用高峰时段优先
    peak_hours: List[tuple] = field(default_factory=list)  # [("07:00", "09:00"), ("17:00", "19:00")]

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
            "peak_hours_enabled": self.peak_hours_enabled,
            "peak_hours": [list(ph) for ph in self.peak_hours],  # 转换为列表以便JSON序列化
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
        segment_duration: int = None,
        peak_hours_enabled: bool = None,
        peak_hours: List[tuple] = None
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
            peak_hours_enabled: 是否启用高峰时段优先（前端传入覆盖配置）
            peak_hours: 高峰时段列表 [("07:00", "09:00"), ("17:00", "19:00")]

        Returns:
            BatchTaskInfo
        """
        batch_id = str(uuid.uuid4())[:8]
        segment_duration = segment_duration or self.history_config.segment_duration

        # 高峰时段配置：前端传入优先，否则使用配置默认值
        if peak_hours_enabled is None:
            peak_hours_enabled = self.batch_config.peak_hours_enabled
        if peak_hours is None:
            peak_hours = list(self.batch_config.default_peak_hours) if peak_hours_enabled else []

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

        # 随机化路口顺序
        if self.batch_config.randomize_road_order:
            road_list = road_list.copy()  # 避免修改原列表
            random.shuffle(road_list)
            logger.info(f"[随机化] 路口顺序已随机打乱，首个路口: {road_list[0] if road_list else 'N/A'}")

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
            peak_hours_enabled=peak_hours_enabled,
            peak_hours=peak_hours,
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

        # 判断遍历模式
        traversal_mode = self.batch_config.traversal_mode

        # 高峰时段遍历模式（优先级最高）
        if task.peak_hours_enabled and task.peak_hours:
            traversal_mode = "peak_hours"
            peak_hours_desc = ", ".join([f"{ph[0]}-{ph[1]}" for ph in task.peak_hours])
            self._log(batch_id, "info",
                      f"批量任务启动 - 高峰时段优先模式, 路口数:{task.total_roads}, "
                      f"时间段:{time_range_desc}, 高峰时段:{peak_hours_desc}")
        else:
            self._log(batch_id, "info",
                      f"批量任务启动 - 模式:{task.mode}, 遍历策略:{traversal_mode}, 路口数:{task.total_roads}, "
                      f"时间段:{time_range_desc}")

        # 发送开始事件
        self._emit_event(BatchEventType.BATCH_START, {
            "batch_id": batch_id,
            "mode": task.mode,
            "traversal_mode": traversal_mode,
            "peak_hours_enabled": task.peak_hours_enabled,
            "peak_hours": [list(ph) for ph in task.peak_hours] if task.peak_hours else [],
            "total_roads": task.total_roads,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "time_range": time_range_desc
        })

        # 构建时间字符串（用于API调用：使用开始日期作为基准）
        # 注：对于跨日期范围，下载器会自动处理分片
        start_str = f"{task.start_date.replace('-', '')}{task.start_time.replace(':', '')}00"
        end_str = f"{task.end_date.replace('-', '')}{task.end_time.replace(':', '')}00"

        if traversal_mode == "peak_hours":
            # 高峰时段优先遍历：三轮处理
            self._peak_hours_traverse(task, start_str, end_str)
        elif traversal_mode == "breadth_first":
            # 广度优先遍历：两轮处理
            self._breadth_first_traverse(task, start_str, end_str)
        else:
            # 深度优先遍历：原有逻辑
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

    def _peak_hours_traverse(
        self,
        task: BatchTaskInfo,
        start_str: str,
        end_str: str
    ):
        """
        高峰时段优先遍历策略：
        第一轮：每个路口1个摄像头，只处理高峰时段
        第1.5轮：检出事故时，立即处理同一路口其他摄像头的事故时段
        第二轮：每个路口1个摄像头，处理非高峰时段
        """
        batch_id = task.batch_id
        cameras_first_pass = self.batch_config.cameras_per_road_first_pass
        peak_hours = task.peak_hours

        peak_hours_desc = ", ".join([f"{ph[0]}-{ph[1]}" for ph in peak_hours])
        self._log(batch_id, "info",
                  f"[高峰优先] 开始遍历 - 高峰时段:{peak_hours_desc}, 每路口{cameras_first_pass}个摄像头")

        # ========== 第一轮：高峰时段快速筛查 ==========
        self._log(batch_id, "info",
                  f"[高峰优先] 第一轮开始 - 遍历所有路口的高峰时段")

        accident_roads = []  # 记录检出事故的路口

        for road_task in task.roads:
            if self._stop_flag.is_set():
                self._log(batch_id, "warning", "批量任务被停止")
                break

            # 第一轮：只处理高峰时段
            result = self._process_road_peak_hours(
                task, road_task, start_str, end_str,
                peak_hours=peak_hours,
                max_cameras=cameras_first_pass,
                pass_number=1
            )

            # 检查是否检出事故
            if result.get("events_found", 0) > 0:
                accident_roads.append(road_task)
                self._log(batch_id, "warning",
                          f"[高峰优先] 路口 {road_task.road_id} 检出事故！")

                # ========== 第1.5轮：立即处理同一路口其他摄像头的事故时段 ==========
                if road_task.accident_time_slots and len(road_task.cameras) > cameras_first_pass:
                    self._log(batch_id, "info",
                              f"[高峰优先] 第1.5轮 - 立即深入分析路口 {road_task.road_id} 的其他摄像头")

                    self._process_road_remaining(
                        task, road_task, start_str, end_str,
                        skip_cameras=cameras_first_pass,
                        pass_number=1.5,
                        priority_time_slots=road_task.accident_time_slots
                    )

        self._log(batch_id, "info",
                  f"[高峰优先] 第一轮完成 - 检出事故路口数: {len(accident_roads)}")

        # ========== 第二轮：非高峰时段 ==========
        self._log(batch_id, "info",
                  f"[高峰优先] 第二轮开始 - 遍历所有路口的非高峰时段")

        for road_task in task.roads:
            if self._stop_flag.is_set():
                self._log(batch_id, "warning", "批量任务被停止")
                break

            # 第二轮：处理非高峰时段
            result = self._process_road_non_peak_hours(
                task, road_task, start_str, end_str,
                peak_hours=peak_hours,
                max_cameras=cameras_first_pass,
                pass_number=2
            )

            # 如果第二轮也检出事故，记录但不立即深入（避免无限扩展）
            if result.get("events_found", 0) > 0:
                if road_task not in accident_roads:
                    accident_roads.append(road_task)
                self._log(batch_id, "warning",
                          f"[高峰优先] 路口 {road_task.road_id} 在非高峰时段也检出事故")

        self._log(batch_id, "info",
                  f"[高峰优先] 第二轮完成 - 总计检出事故路口数: {len(accident_roads)}")

        # 发送高峰时段遍历统计
        self._emit_event(BatchEventType.BATCH_PROGRESS, {
            "batch_id": batch_id,
            "traversal_mode": "peak_hours",
            "first_pass_roads": task.total_roads,
            "accident_roads": len(accident_roads),
            "peak_hours": [list(ph) for ph in peak_hours]
        })

    def _process_road_peak_hours(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        start_str: str,
        end_str: str,
        peak_hours: List[tuple],
        max_cameras: int = 1,
        pass_number: int = 1
    ) -> Dict[str, Any]:
        """
        处理路口的高峰时段

        Args:
            peak_hours: 高峰时段列表 [("07:00", "09:00"), ("17:00", "19:00")]
            max_cameras: 最多处理的摄像头数
            pass_number: 第几轮遍历

        Returns:
            {"events_found": int, "cameras_processed": int}
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id

        road_task.status = RoadStatus.RUNNING
        peak_hours_desc = ", ".join([f"{ph[0]}-{ph[1]}" for ph in peak_hours])

        self._log(batch_id, "info",
                  f"[Pass {pass_number}] 处理路口 {road_id} 高峰时段: {peak_hours_desc}",
                  category="road")

        self._emit_event(BatchEventType.ROAD_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "road_name": road_task.road_name,
            "road_index": batch_task.roads.index(road_task),
            "total_roads": batch_task.total_roads,
            "pass_number": pass_number,
            "time_type": "peak_hours"
        })

        result = {"events_found": 0, "cameras_processed": 0}

        try:
            # 获取全景摄像头列表
            cameras = self._get_panoramic_cameras(road_id, start_str, end_str)

            if not cameras:
                road_task.status = RoadStatus.SKIPPED
                road_task.error_message = "无可用全景摄像头"
                batch_task.skipped_roads += 1

                self._emit_event(BatchEventType.ROAD_SKIPPED, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "reason": "无可用全景摄像头"
                })
                return result

            # 随机选择摄像头
            if self.batch_config.randomize_camera_selection and len(cameras) > max_cameras:
                cameras_to_process = random.sample(cameras, max_cameras)
                logger.info(f"[随机化] 路口 {road_id} 随机选择摄像头: {[c.channel_num for c in cameras_to_process]}")
            else:
                cameras_to_process = cameras[:max_cameras]

            road_task.total_cameras = len(cameras)

            # 创建摄像头任务（如果尚未创建）
            if not road_task.cameras:
                for cam in cameras:
                    road_task.cameras.append(CameraTask(camera_info=cam))

            # 找到选中的摄像头任务
            selected_channel_nums = {c.channel_num for c in cameras_to_process}

            # 处理选中的摄像头（只处理高峰时段）
            for camera_task in road_task.cameras:
                if camera_task.camera_info.channel_num not in selected_channel_nums:
                    continue
                if self._stop_flag.is_set():
                    break

                try:
                    # 为每个高峰时段创建分析任务
                    for peak_start, peak_end in peak_hours:
                        self._process_camera_time_range(
                            batch_task, road_task, camera_task,
                            time_start=peak_start,
                            time_end=peak_end,
                            time_type="peak"
                        )

                    result["cameras_processed"] += 1
                    result["events_found"] += camera_task.events_found

                except Exception as e:
                    camera_task.status = CameraStatus.FAILED
                    camera_task.error_message = str(e)
                    road_task.skipped_cameras += 1

            # 更新路口状态
            road_task.status = RoadStatus.RUNNING  # 高峰时段完成，还有非高峰时段

            self._emit_event(BatchEventType.ROAD_PROGRESS, {
                "batch_id": batch_id,
                "road_id": road_id,
                "pass_number": pass_number,
                "cameras_processed": result["cameras_processed"],
                "events_found": result["events_found"],
                "time_type": "peak_hours"
            })

        except Exception as e:
            road_task.status = RoadStatus.FAILED
            road_task.error_message = str(e)
            batch_task.skipped_roads += 1

            self._emit_event(BatchEventType.ROAD_SKIPPED, {
                "batch_id": batch_id,
                "road_id": road_id,
                "reason": str(e)
            })

        return result

    def _process_road_non_peak_hours(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        start_str: str,
        end_str: str,
        peak_hours: List[tuple],
        max_cameras: int = 1,
        pass_number: int = 2
    ) -> Dict[str, Any]:
        """
        处理路口的非高峰时段

        Args:
            peak_hours: 高峰时段列表（用于排除）
            max_cameras: 最多处理的摄像头数
            pass_number: 第几轮遍历

        Returns:
            {"events_found": int, "cameras_processed": int}
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id

        # 计算非高峰时段
        non_peak_slots = self._calculate_non_peak_slots(
            batch_task.start_time, batch_task.end_time, peak_hours
        )

        if not non_peak_slots:
            self._log(batch_id, "info",
                      f"[Pass {pass_number}] 路口 {road_id} 无非高峰时段需处理",
                      category="road")
            return {"events_found": 0, "cameras_processed": 0}

        non_peak_desc = ", ".join([f"{s[0]}-{s[1]}" for s in non_peak_slots])
        self._log(batch_id, "info",
                  f"[Pass {pass_number}] 处理路口 {road_id} 非高峰时段: {non_peak_desc}",
                  category="road")

        self._emit_event(BatchEventType.ROAD_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "road_name": road_task.road_name,
            "road_index": batch_task.roads.index(road_task),
            "total_roads": batch_task.total_roads,
            "pass_number": pass_number,
            "time_type": "non_peak_hours"
        })

        result = {"events_found": 0, "cameras_processed": 0}

        try:
            # 复用第一轮已选择的摄像头
            if not road_task.cameras:
                self._log(batch_id, "warning",
                          f"路口 {road_id} 无摄像头任务，跳过非高峰时段",
                          category="road")
                return result

            # 找到第一轮处理过的摄像头（状态非PENDING）
            processed_cameras = [c for c in road_task.cameras
                                 if c.status not in [CameraStatus.PENDING]][:max_cameras]

            if not processed_cameras:
                # 如果没有处理过的，随机选择
                if self.batch_config.randomize_camera_selection and len(road_task.cameras) > max_cameras:
                    processed_cameras = random.sample(road_task.cameras, max_cameras)
                else:
                    processed_cameras = road_task.cameras[:max_cameras]

            # 处理非高峰时段
            for camera_task in processed_cameras:
                if self._stop_flag.is_set():
                    break

                try:
                    # 为每个非高峰时段创建分析任务
                    for slot_start, slot_end in non_peak_slots:
                        self._process_camera_time_range(
                            batch_task, road_task, camera_task,
                            time_start=slot_start,
                            time_end=slot_end,
                            time_type="non_peak"
                        )

                    result["cameras_processed"] += 1
                    result["events_found"] += camera_task.events_found

                except Exception as e:
                    self._log(batch_id, "error",
                              f"摄像头 {camera_task.camera_info.channel_num} 非高峰时段处理失败: {e}",
                              category="camera")

            # 更新路口状态
            road_task.status = RoadStatus.COMPLETED
            batch_task.completed_roads += 1

            self._emit_event(BatchEventType.ROAD_COMPLETE, {
                "batch_id": batch_id,
                "road_id": road_id,
                "pass_number": pass_number,
                "total_cameras": road_task.total_cameras,
                "completed_cameras": road_task.completed_cameras,
                "events_found": road_task.events_found
            })

        except Exception as e:
            self._log(batch_id, "error",
                      f"路口 {road_id} 非高峰时段处理异常: {e}",
                      category="road")

        return result

    def _calculate_non_peak_slots(
        self,
        start_time: str,
        end_time: str,
        peak_hours: List[tuple]
    ) -> List[tuple]:
        """
        计算非高峰时段

        Args:
            start_time: 任务开始时间 "00:00"
            end_time: 任务结束时间 "23:59"
            peak_hours: 高峰时段列表 [("07:00", "09:00"), ("17:00", "19:00")]

        Returns:
            非高峰时段列表 [("00:00", "07:00"), ("09:00", "17:00"), ("19:00", "23:59")]
        """
        def time_to_minutes(t: str) -> int:
            h, m = map(int, t.split(":"))
            return h * 60 + m

        def minutes_to_time(m: int) -> str:
            return f"{m // 60:02d}:{m % 60:02d}"

        start_min = time_to_minutes(start_time)
        end_min = time_to_minutes(end_time)

        # 如果结束时间小于开始时间，说明跨天
        if end_min <= start_min:
            end_min += 24 * 60

        # 将高峰时段转换为分钟并排序
        peak_slots = []
        for ps, pe in peak_hours:
            ps_min = time_to_minutes(ps)
            pe_min = time_to_minutes(pe)
            # 只包含在任务时间范围内的高峰时段
            if ps_min < end_min and pe_min > start_min:
                peak_slots.append((max(ps_min, start_min), min(pe_min, end_min)))

        peak_slots.sort()

        # 计算非高峰时段
        non_peak_slots = []
        current = start_min

        for ps_min, pe_min in peak_slots:
            if current < ps_min:
                # 当前位置到高峰开始是非高峰时段
                non_peak_slots.append((
                    minutes_to_time(current % (24 * 60)),
                    minutes_to_time(ps_min % (24 * 60))
                ))
            current = max(current, pe_min)

        # 最后一个高峰结束到任务结束
        if current < end_min:
            non_peak_slots.append((
                minutes_to_time(current % (24 * 60)),
                minutes_to_time(end_min % (24 * 60))
            ))

        return non_peak_slots

    def _process_camera_time_range(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        camera_task: CameraTask,
        time_start: str,
        time_end: str,
        time_type: str = "full"
    ):
        """
        处理摄像头的指定时间范围

        Args:
            time_start: 开始时间 "07:00"
            time_end: 结束时间 "09:00"
            time_type: "peak" | "non_peak" | "full"
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id
        channel_num = camera_task.camera_info.channel_num

        camera_task.status = CameraStatus.RUNNING

        self._log(batch_id, "info",
                  f"[{time_type}] 处理摄像头 {channel_num} (路口 {road_id}) 时段 {time_start}-{time_end}",
                  category="camera")

        # 发送摄像头开始事件
        self._emit_event(BatchEventType.CAMERA_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "camera_index": road_task.cameras.index(camera_task),
            "total_cameras": road_task.total_cameras,
            "time_range": f"{time_start}-{time_end}",
            "time_type": time_type
        })

        # 创建事件收集器
        events_found = 0
        events_cleared = 0
        accident_slots = []

        def camera_event_callback(event_type: EventType, data: dict):
            """摄像头事件回调"""
            nonlocal events_found, events_cleared

            if event_type == EventType.RESULT:
                events_found += 1

                # 提取事故时段信息
                segment_info = data.get("segment_info", {})
                result_data = data.get("result", {})

                accidents = result_data.get("accidents", [])
                if accidents or result_data.get("has_accident"):
                    slot = AccidentTimeSlot(
                        start_time=segment_info.get("start_time", time_start),
                        end_time=segment_info.get("end_time", time_end),
                        segment_index=segment_info.get("index", 0),
                        confidence=max([a.get("confidence", 0.5) for a in accidents]) if accidents else 0.5,
                        camera_channel=channel_num
                    )
                    accident_slots.append(slot)
                    logger.info(f"[事故时段] 检出事故: 路口{road_id} 摄像头{channel_num} "
                               f"时段 {slot.start_time}-{slot.end_time}")

                # 转发结果事件
                self._emit_event(BatchEventType.RESULT, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "channel_num": channel_num,
                    "time_type": time_type,
                    **data
                })

            elif event_type == EventType.QUEUE:
                camera_task.total_segments = data.get("total", 0)
                camera_task.completed_segments = data.get("completed", 0)

            elif event_type == EventType.LOG:
                self._emit_event(BatchEventType.LOG, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "channel_num": channel_num,
                    **data
                })

            elif event_type == EventType.COMPLETE:
                events_cleared = data.get("events_cleared", 0)

        # 创建 HistoryVideoProcessor 处理指定时间范围
        processor = HistoryVideoProcessor(
            api=self.api,
            config=self.history_config,
            pipeline_func=self.pipeline_func,
            event_callback=camera_event_callback
        )

        # 创建任务（使用指定的时间范围）
        task = processor.create_task(
            road_id=road_id,
            channel_num=channel_num,
            start_date=batch_task.start_date,
            start_time=time_start,
            end_date=batch_task.start_date,  # 假设同一天
            end_time=time_end,
            mode=batch_task.analysis_mode,
            model=batch_task.model,
            violation_types=batch_task.violation_types,
            segment_duration=batch_task.segment_duration
        )

        camera_task.task_id = task.task_id

        # 启动任务（阻塞直到完成）
        processor.start_task(task.task_id)

        # 更新统计
        camera_task.events_found += events_found
        camera_task.events_cleared += events_cleared
        camera_task.status = CameraStatus.COMPLETED

        road_task.events_found += events_found
        road_task.events_cleared += events_cleared
        road_task.completed_cameras += 1

        # 记录事故时段到路口任务
        if accident_slots:
            road_task.accident_time_slots.extend(accident_slots)

        batch_task.total_events_found += events_found
        batch_task.total_events_cleared += events_cleared
        batch_task.completed_cameras += 1

        self._log(batch_id, "success",
                  f"[{time_type}] 摄像头 {channel_num} 时段 {time_start}-{time_end} 完成 - 检出:{events_found}",
                  category="camera")

        # 发送摄像头完成事件
        self._emit_event(BatchEventType.CAMERA_COMPLETE, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "events_found": events_found,
            "time_range": f"{time_start}-{time_end}",
            "time_type": time_type
        })

    def _breadth_first_traverse(
        self,
        task: BatchTaskInfo,
        start_str: str,
        end_str: str
    ):
        """
        广度优先遍历策略：
        第一轮：每个路口快速筛查（只处理1个摄像头）
        第二轮：对检出事故的路口深入分析（处理剩余摄像头）
        """
        batch_id = task.batch_id
        cameras_first_pass = self.batch_config.cameras_per_road_first_pass

        self._log(batch_id, "info",
                  f"[广度优先] 第一轮快筛开始 - 每路口处理{cameras_first_pass}个摄像头")

        # ========== 第一轮：快速筛查 ==========
        accident_roads = []  # 记录检出事故的路口

        for road_task in task.roads:
            if self._stop_flag.is_set():
                self._log(batch_id, "warning", "批量任务被停止")
                break

            # 第一轮只处理有限数量的摄像头
            result = self._process_road_limited(
                task, road_task, start_str, end_str,
                max_cameras=cameras_first_pass,
                pass_number=1
            )

            # 检查是否检出事故
            if result.get("events_found", 0) > 0:
                accident_roads.append(road_task)
                self._log(batch_id, "warning",
                          f"[广度优先] 路口 {road_task.road_id} 检出事故，加入深入分析队列")

        self._log(batch_id, "info",
                  f"[广度优先] 第一轮完成 - 检出事故路口数: {len(accident_roads)}")

        # ========== 第二轮：深入分析 ==========
        if self.batch_config.second_pass_enabled and accident_roads:
            self._log(batch_id, "info",
                      f"[广度优先] 第二轮深入分析开始 - {len(accident_roads)}个路口")

            # 按事故严重程度排序（事故数多的优先，置信度高的优先）
            if self.batch_config.prioritize_accident_roads:
                accident_roads.sort(
                    key=lambda r: (
                        r.events_found,
                        max([s.confidence for s in r.accident_time_slots], default=0)
                    ),
                    reverse=True
                )

            for road_task in accident_roads:
                if self._stop_flag.is_set():
                    break

                # 获取该路口的事故时段
                priority_slots = road_task.accident_time_slots

                # 第二轮处理剩余摄像头
                remaining_cameras = len(road_task.cameras) - cameras_first_pass
                if remaining_cameras > 0:
                    if priority_slots:
                        # 显示要优先分析的事故时段
                        slot_desc = ", ".join([f"{s.start_time}-{s.end_time}" for s in priority_slots[:3]])
                        self._log(batch_id, "info",
                                  f"[广度优先] 深入分析路口 {road_task.road_id}，"
                                  f"剩余{remaining_cameras}个摄像头，优先时段: {slot_desc}")
                    else:
                        self._log(batch_id, "info",
                                  f"[广度优先] 深入分析路口 {road_task.road_id}，剩余{remaining_cameras}个摄像头")

                    self._process_road_remaining(
                        task, road_task, start_str, end_str,
                        skip_cameras=cameras_first_pass,
                        pass_number=2,
                        priority_time_slots=priority_slots
                    )

            self._log(batch_id, "info",
                      f"[广度优先] 第二轮深入分析完成")

        # 发送广度优先遍历统计
        self._emit_event(BatchEventType.BATCH_PROGRESS, {
            "batch_id": batch_id,
            "traversal_mode": "breadth_first",
            "first_pass_roads": task.total_roads,
            "accident_roads": len(accident_roads),
            "second_pass_roads": len(accident_roads) if self.batch_config.second_pass_enabled else 0
        })

    def _process_road_limited(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        start_str: str,
        end_str: str,
        max_cameras: int = 1,
        pass_number: int = 1
    ) -> Dict[str, Any]:
        """
        处理路口（限制摄像头数量）

        Args:
            max_cameras: 最多处理的摄像头数
            pass_number: 第几轮遍历

        Returns:
            {"events_found": int, "cameras_processed": int}
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id

        road_task.status = RoadStatus.RUNNING

        self._log(batch_id, "info",
                  f"[Pass {pass_number}] 开始处理路口 {road_id}", category="road")

        self._emit_event(BatchEventType.ROAD_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "road_name": road_task.road_name,
            "road_index": batch_task.roads.index(road_task),
            "total_roads": batch_task.total_roads,
            "pass_number": pass_number
        })

        result = {"events_found": 0, "cameras_processed": 0}

        try:
            # 获取全景摄像头列表
            cameras = self._get_panoramic_cameras(road_id, start_str, end_str)

            if not cameras:
                road_task.status = RoadStatus.SKIPPED
                road_task.error_message = "无可用全景摄像头"
                batch_task.skipped_roads += 1

                self._emit_event(BatchEventType.ROAD_SKIPPED, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "reason": "无可用全景摄像头"
                })
                return result

            # 随机选择摄像头
            if self.batch_config.randomize_camera_selection and len(cameras) > max_cameras:
                cameras_to_process = random.sample(cameras, max_cameras)
                logger.info(f"[随机化] 路口 {road_id} 随机选择摄像头: {[c.channel_num for c in cameras_to_process]}")
            else:
                cameras_to_process = cameras[:max_cameras]

            road_task.total_cameras = len(cameras)

            # 创建摄像头任务（如果尚未创建）
            if not road_task.cameras:
                for cam in cameras:
                    road_task.cameras.append(CameraTask(camera_info=cam))

            # 找到选中的摄像头任务
            selected_channel_nums = {c.channel_num for c in cameras_to_process}

            # 处理限定数量的摄像头
            for camera_task in road_task.cameras:
                if camera_task.camera_info.channel_num not in selected_channel_nums:
                    continue
                if self._stop_flag.is_set():
                    break

                try:
                    self._process_camera(batch_task, road_task, camera_task)
                    result["cameras_processed"] += 1
                    result["events_found"] += camera_task.events_found

                except Exception as e:
                    camera_task.status = CameraStatus.FAILED
                    camera_task.error_message = str(e)
                    road_task.skipped_cameras += 1

            # 更新路口状态（第一轮不标记完成，留待第二轮）
            if pass_number == 1 and max_cameras < len(cameras):
                road_task.status = RoadStatus.RUNNING  # 保持运行状态
            else:
                road_task.status = RoadStatus.COMPLETED
                batch_task.completed_roads += 1

            self._emit_event(BatchEventType.ROAD_PROGRESS, {
                "batch_id": batch_id,
                "road_id": road_id,
                "pass_number": pass_number,
                "cameras_processed": result["cameras_processed"],
                "events_found": result["events_found"]
            })

        except Exception as e:
            road_task.status = RoadStatus.FAILED
            road_task.error_message = str(e)
            batch_task.skipped_roads += 1

            self._emit_event(BatchEventType.ROAD_SKIPPED, {
                "batch_id": batch_id,
                "road_id": road_id,
                "reason": str(e)
            })

        return result

    def _process_road_remaining(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        start_str: str,
        end_str: str,
        skip_cameras: int = 1,
        pass_number: int = 2,
        priority_time_slots: List[AccidentTimeSlot] = None
    ):
        """
        处理路口剩余摄像头（第二轮深入分析）

        Args:
            skip_cameras: 跳过前N个摄像头（已在第一轮处理）
            pass_number: 第几轮遍历
            priority_time_slots: 优先分析的事故时段列表
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id

        # 如果有优先时段，只分析这些时段
        if priority_time_slots:
            # 合并相近的时段，避免重复分析
            merged_slots = self._merge_time_slots(priority_time_slots)
            slot_desc = ", ".join([f"{s.start_time}-{s.end_time}" for s in merged_slots[:3]])
            self._log(batch_id, "info",
                      f"[Pass {pass_number}] 深入分析路口 {road_id}，"
                      f"优先时段: {slot_desc}，跳过前{skip_cameras}个摄像头",
                      category="road")
        else:
            self._log(batch_id, "info",
                      f"[Pass {pass_number}] 深入分析路口 {road_id}，跳过前{skip_cameras}个摄像头",
                      category="road")

        # 处理剩余摄像头
        for camera_task in road_task.cameras[skip_cameras:]:
            if self._stop_flag.is_set():
                break

            if camera_task.status != CameraStatus.PENDING:
                continue  # 跳过已处理的

            try:
                # 如果有优先时段，使用优先时段的时间范围进行分析
                if priority_time_slots:
                    self._process_camera_with_priority_slots(
                        batch_task, road_task, camera_task, priority_time_slots
                    )
                else:
                    self._process_camera(batch_task, road_task, camera_task)

            except Exception as e:
                camera_task.status = CameraStatus.FAILED
                camera_task.error_message = str(e)
                road_task.skipped_cameras += 1

                self._log(batch_id, "error",
                          f"摄像头 {camera_task.camera_info.channel_num} 处理失败: {e}",
                          category="camera")

        # 标记路口完成
        road_task.status = RoadStatus.COMPLETED

        self._emit_event(BatchEventType.ROAD_COMPLETE, {
            "batch_id": batch_id,
            "road_id": road_id,
            "pass_number": pass_number,
            "total_cameras": road_task.total_cameras,
            "completed_cameras": road_task.completed_cameras,
            "events_found": road_task.events_found
        })

    def _merge_time_slots(self, slots: List[AccidentTimeSlot]) -> List[AccidentTimeSlot]:
        """合并相近的时间段"""
        if not slots:
            return []

        # 按开始时间排序
        sorted_slots = sorted(slots, key=lambda s: s.start_time)
        merged = [sorted_slots[0]]

        for slot in sorted_slots[1:]:
            last = merged[-1]
            # 如果时段重叠或相邻，合并
            if slot.start_time <= last.end_time:
                # 扩展结束时间
                if slot.end_time > last.end_time:
                    merged[-1] = AccidentTimeSlot(
                        start_time=last.start_time,
                        end_time=slot.end_time,
                        segment_index=last.segment_index,
                        confidence=max(last.confidence, slot.confidence),
                        camera_channel=last.camera_channel
                    )
            else:
                merged.append(slot)

        return merged

    def _process_camera_with_priority_slots(
        self,
        batch_task: BatchTaskInfo,
        road_task: RoadTask,
        camera_task: CameraTask,
        priority_slots: List[AccidentTimeSlot]
    ):
        """
        使用优先时段处理摄像头（只分析事故时段附近）

        优化策略：第二轮只分析检出事故的时段，而不是整个时间范围
        """
        batch_id = batch_task.batch_id
        road_id = road_task.road_id
        channel_num = camera_task.camera_info.channel_num

        # 合并时段
        merged_slots = self._merge_time_slots(priority_slots)

        camera_task.status = CameraStatus.RUNNING

        self._log(batch_id, "info",
                  f"[优先时段分析] 摄像头 {channel_num} (路口 {road_id})，"
                  f"只分析 {len(merged_slots)} 个事故时段",
                  category="camera")

        # 发送摄像头开始事件
        self._emit_event(BatchEventType.CAMERA_START, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "camera_index": road_task.cameras.index(camera_task),
            "total_cameras": road_task.total_cameras,
            "priority_analysis": True,
            "time_slots": [{"start": s.start_time, "end": s.end_time} for s in merged_slots]
        })

        events_found = 0
        events_cleared = 0
        accident_slots = []

        def camera_event_callback(event_type: EventType, data: dict):
            """摄像头事件回调"""
            nonlocal events_found, events_cleared

            if event_type == EventType.RESULT:
                events_found += 1
                segment_info = data.get("segment_info", {})
                result_data = data.get("result", {})

                accidents = result_data.get("accidents", [])
                if accidents or result_data.get("has_accident"):
                    slot = AccidentTimeSlot(
                        start_time=segment_info.get("start_time", ""),
                        end_time=segment_info.get("end_time", ""),
                        segment_index=segment_info.get("index", 0),
                        confidence=max([a.get("confidence", 0.5) for a in accidents]) if accidents else 0.5,
                        camera_channel=channel_num
                    )
                    accident_slots.append(slot)

                self._emit_event(BatchEventType.RESULT, {
                    "batch_id": batch_id,
                    "road_id": road_id,
                    "channel_num": channel_num,
                    "priority_analysis": True,
                    **data
                })

            elif event_type == EventType.COMPLETE:
                events_cleared = data.get("events_cleared", 0)

        # 对每个优先时段创建分析任务
        for slot in merged_slots:
            if self._stop_flag.is_set():
                break

            # 使用优先时段的时间范围
            processor = HistoryVideoProcessor(
                api=self.api,
                config=self.history_config,
                pipeline_func=self.pipeline_func,
                event_callback=camera_event_callback
            )

            # 解析时间（假设格式为 HH:MM 或完整日期时间）
            slot_start_time = slot.start_time
            slot_end_time = slot.end_time

            # 如果只有时间部分，使用批量任务的日期
            if len(slot_start_time) <= 5:  # HH:MM格式
                slot_start_date = batch_task.start_date
                slot_end_date = batch_task.start_date
            else:
                # 假设是完整时间，提取日期
                slot_start_date = slot_start_time[:10] if len(slot_start_time) >= 10 else batch_task.start_date
                slot_end_date = slot_end_time[:10] if len(slot_end_time) >= 10 else batch_task.end_date
                slot_start_time = slot_start_time[-5:] if len(slot_start_time) > 5 else slot_start_time
                slot_end_time = slot_end_time[-5:] if len(slot_end_time) > 5 else slot_end_time

            try:
                task = processor.create_task(
                    road_id=road_id,
                    channel_num=channel_num,
                    start_date=slot_start_date,
                    start_time=slot_start_time,
                    end_date=slot_end_date,
                    end_time=slot_end_time,
                    mode=batch_task.analysis_mode,
                    model=batch_task.model,
                    violation_types=batch_task.violation_types,
                    segment_duration=batch_task.segment_duration
                )

                processor.start_task(task.task_id)

            except Exception as e:
                self._log(batch_id, "warning",
                          f"优先时段 {slot_start_time}-{slot_end_time} 分析失败: {e}",
                          category="camera")

        # 更新统计
        camera_task.events_found = events_found
        camera_task.events_cleared = events_cleared
        camera_task.status = CameraStatus.COMPLETED

        road_task.events_found += events_found
        road_task.events_cleared += events_cleared
        road_task.completed_cameras += 1

        if accident_slots:
            road_task.accident_time_slots.extend(accident_slots)

        batch_task.total_events_found += events_found
        batch_task.total_events_cleared += events_cleared
        batch_task.completed_cameras += 1

        self._log(batch_id, "success",
                  f"[优先时段分析] 摄像头 {channel_num} 完成 - 检出:{events_found}",
                  category="camera")

        self._emit_event(BatchEventType.CAMERA_COMPLETE, {
            "batch_id": batch_id,
            "road_id": road_id,
            "channel_num": channel_num,
            "events_found": events_found,
            "priority_analysis": True
        })

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
        accident_slots = []  # 记录检出的事故时段

        def camera_event_callback(event_type: EventType, data: dict):
            """摄像头事件回调，转发到批量事件"""
            nonlocal events_found, events_cleared

            # 统计结果
            if event_type == EventType.RESULT:
                events_found += 1

                # 提取事故时段信息
                segment_info = data.get("segment_info", {})
                result_data = data.get("result", {})

                # 从结果中提取事故信息
                accidents = result_data.get("accidents", [])
                if accidents or result_data.get("has_accident"):
                    # 记录事故时段
                    slot = AccidentTimeSlot(
                        start_time=segment_info.get("start_time", batch_task.start_time),
                        end_time=segment_info.get("end_time", batch_task.end_time),
                        segment_index=segment_info.get("index", 0),
                        confidence=max([a.get("confidence", 0.5) for a in accidents]) if accidents else 0.5,
                        camera_channel=channel_num
                    )
                    accident_slots.append(slot)
                    logger.info(f"[事故时段] 检出事故: 路口{road_id} 摄像头{channel_num} "
                               f"时段 {slot.start_time}-{slot.end_time}")

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

        # P0优化：记录事故时段到路口任务
        if accident_slots:
            road_task.accident_time_slots.extend(accident_slots)
            self._log(batch_id, "info",
                      f"路口 {road_id} 记录 {len(accident_slots)} 个事故时段",
                      category="road")

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
