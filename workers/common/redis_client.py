"""
Redis Streams 客户端封装
支持 video_tasks 和 result_tasks 两个核心队列
"""
import os
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis


@dataclass
class VideoTask:
    """video_tasks 消息结构"""
    job_id: str
    camera_id: str
    window_path: str
    trace_id: str
    seg_end_ts: int = None  # 可选字段
    created_at: float = None
    analysis_type: str = "accident"  # 新增: 分析类型 (accident/mv_violation/ebike_violation/ads_behavior)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "VideoTask":
        seg_end_ts_raw = data.get("seg_end_ts")
        seg_end_ts = None
        if seg_end_ts_raw and seg_end_ts_raw not in ("None", "null", ""):
            seg_end_ts = int(seg_end_ts_raw)
        return cls(
            job_id=data["job_id"],
            camera_id=data["camera_id"],
            window_path=data["window_path"],
            seg_end_ts=seg_end_ts,
            trace_id=data["trace_id"],
            created_at=float(data.get("created_at", time.time())),
            analysis_type=data.get("analysis_type", "accident")  # 默认为事故检测
        )


@dataclass
class EdgeEvent:
    """edge_events 消息结构 - 边缘触发事件"""
    camera_id: str
    timestamp: int  # Unix timestamp
    similarity_score: float
    keyframe_count: int
    keyframes_base64: List[str]  # 关键帧 base64 编码
    window_path: str
    trigger_time: str  # ISO 格式时间
    trace_id: str = ""
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if not self.trace_id:
            self.trace_id = f"{self.camera_id}_{self.timestamp}"

    def to_dict(self) -> Dict[str, str]:
        d = asdict(self)
        d["keyframes_base64"] = json.dumps(d["keyframes_base64"])  # JSON 序列化列表
        return {k: str(v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EdgeEvent":
        keyframes = data.get("keyframes_base64", "[]")
        if isinstance(keyframes, str):
            keyframes = json.loads(keyframes)
        return cls(
            camera_id=data["camera_id"],
            timestamp=int(data["timestamp"]),
            similarity_score=float(data["similarity_score"]),
            keyframe_count=int(data["keyframe_count"]),
            keyframes_base64=keyframes,
            window_path=data["window_path"],
            trigger_time=data["trigger_time"],
            trace_id=data.get("trace_id", ""),
            created_at=float(data.get("created_at", time.time()))
        )


@dataclass
class ResultTask:
    """result_tasks 消息结构"""
    job_id: str
    camera_id: str
    event_time: float
    confidence: float
    result_path: str
    trace_id: str
    is_accident: bool
    seg_end_ts: int = None  # 可选字段
    created_at: float = None
    analysis_type: str = "accident"  # 新增: 分析类型
    is_positive: bool = False  # 新增: 通用阳性标志（违法/事故/异常行为）
    violation_type: str = None  # 新增: 违法类型 (用于 mv_violation/ebike_violation)
    behavior_type: str = None  # 新增: 行为类型 (用于 ads_behavior)
    processing_time_sec: float = 0.0  # 实际处理耗时(秒)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, str]:
        d = asdict(self)
        d["is_accident"] = "1" if self.is_accident else "0"
        d["is_positive"] = "1" if self.is_positive else "0"
        return {k: str(v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "ResultTask":
        return cls(
            job_id=data["job_id"],
            camera_id=data["camera_id"],
            event_time=float(data["event_time"]),
            confidence=float(data["confidence"]),
            result_path=data["result_path"],
            trace_id=data["trace_id"],
            is_accident=data.get("is_accident", "0") == "1",
            created_at=float(data.get("created_at", time.time())),
            analysis_type=data.get("analysis_type", "accident"),
            is_positive=data.get("is_positive", "0") == "1",
            violation_type=data.get("violation_type") if data.get("violation_type") != "None" else None,
            behavior_type=data.get("behavior_type") if data.get("behavior_type") != "None" else None,
            processing_time_sec=float(data.get("processing_time_sec", 0.0)),
        )


class RedisStreamClient:
    """Redis Streams 客户端"""

    STREAM_VIDEO_TASKS = "video_tasks"
    STREAM_RESULT_TASKS = "result_tasks"
    STREAM_EDGE_EVENTS = "edge_events"  # 边缘触发事件队列
    STREAM_DLQ = "dlq_tasks"

    MAX_STREAM_LEN = 0  # 0 = no limit
    STATUS_TTL = 7 * 24 * 3600  # 7 days

    def __init__(
        self,
        host: str = None,
        port: int = 6379,
        db: int = 0,
        password: str = None
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port
        self.db = db
        self.password = password
        self._client = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
        return self._client

    def ensure_consumer_group(self, stream: str, group: str):
        """确保 Consumer Group 存在"""
        try:
            self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    # === video_tasks 操作 ===

    def add_video_task(self, task: VideoTask) -> str:
        """添加视频任务到队列"""
        msg_id = self.client.xadd(
            self.STREAM_VIDEO_TASKS,
            task.to_dict(),
            maxlen=self.MAX_STREAM_LEN or None
        )
        # 更新任务状态
        self.set_task_status(task.job_id, "pending")
        return msg_id

    def read_video_tasks(
        self,
        group: str = None,
        consumer: str = None,
        count: int = 1,
        block: int = 5000,
        consumer_group: str = None,
        consumer_name: str = None,
        block_ms: int = None
    ) -> List[tuple]:
        # 兼容两种参数命名
        group = group or consumer_group
        consumer = consumer or consumer_name
        block = block_ms if block_ms is not None else block
        """读取视频任务（阻塞模式）先读pending再读新消息"""
        self.ensure_consumer_group(self.STREAM_VIDEO_TASKS, group)
        # 先尝试读取pending消息 (未ACK的)
        result = self.client.xreadgroup(
            group,
            consumer,
            {self.STREAM_VIDEO_TASKS: "0"},  # 读取pending
            count=count,
            block=0  # 不阻塞
        )
        # 如果没有pending，再读新消息
        if not result or not result[0][1]:
            result = self.client.xreadgroup(
                group,
                consumer,
                {self.STREAM_VIDEO_TASKS: ">"},  # 读取新消息
                count=count,
                block=block
            )
        if not result:
            return []
        # 解析结果
        tasks = []
        for stream_name, messages in result:
            for msg_id, data in messages:
                tasks.append((msg_id, VideoTask.from_dict(data)))
        return tasks

    def ack_video_task(self, group: str, msg_id: str):
        """确认视频任务处理完成"""
        self.client.xack(self.STREAM_VIDEO_TASKS, group, msg_id)

    # === result_tasks 操作 ===

    def add_result_task(self, task: ResultTask) -> str:
        """添加结果任务到队列"""
        msg_id = self.client.xadd(
            self.STREAM_RESULT_TASKS,
            task.to_dict(),
            maxlen=self.MAX_STREAM_LEN or None
        )
        return msg_id

    def read_result_tasks(
        self,
        group: str,
        consumer: str,
        count: int = 1,
        block: int = 5000
    ) -> List[tuple]:
        """读取结果任务"""
        self.ensure_consumer_group(self.STREAM_RESULT_TASKS, group)
        result = self.client.xreadgroup(
            group,
            consumer,
            {self.STREAM_RESULT_TASKS: ">"},
            count=count,
            block=block
        )
        if not result:
            return []
        tasks = []
        for stream_name, messages in result:
            for msg_id, data in messages:
                tasks.append((msg_id, ResultTask.from_dict(data)))
        return tasks

    def ack_result_task(self, group: str, msg_id: str):
        """确认结果任务处理完成"""
        self.client.xack(self.STREAM_RESULT_TASKS, group, msg_id)

    # === edge_events 操作 ===

    def add_edge_event(self, event: EdgeEvent) -> str:
        """添加边缘触发事件到队列"""
        msg_id = self.client.xadd(
            self.STREAM_EDGE_EVENTS,
            event.to_dict(),
            maxlen=self.MAX_STREAM_LEN or None
        )
        return msg_id

    def read_edge_events(
        self,
        group: str,
        consumer: str,
        count: int = 1,
        block: int = 5000
    ) -> List[tuple]:
        """读取边缘触发事件"""
        self.ensure_consumer_group(self.STREAM_EDGE_EVENTS, group)
        # 先尝试读取pending消息
        result = self.client.xreadgroup(
            group,
            consumer,
            {self.STREAM_EDGE_EVENTS: "0"},
            count=count,
            block=0
        )
        # 如果没有pending，再读新消息
        if not result or not result[0][1]:
            result = self.client.xreadgroup(
                group,
                consumer,
                {self.STREAM_EDGE_EVENTS: ">"},
                count=count,
                block=block
            )
        if not result:
            return []
        tasks = []
        for stream_name, messages in result:
            for msg_id, data in messages:
                tasks.append((msg_id, EdgeEvent.from_dict(data)))
        return tasks

    def ack_edge_event(self, group: str, msg_id: str):
        """确认边缘事件处理完成"""
        self.client.xack(self.STREAM_EDGE_EVENTS, group, msg_id)

    # === 任务状态管理 ===

    def set_task_status(
        self,
        job_id: str,
        state: str,
        worker_id: str = None,
        extra: Dict = None
    ):
        """设置任务状态"""
        key = f"task_status:{job_id}"
        data = {
            "state": state,
            "updated_at": time.time()
        }
        if worker_id:
            data["worker_id"] = worker_id
        if extra:
            data.update(extra)
        self.client.hset(key, mapping={k: str(v) for k, v in data.items()})
        self.client.expire(key, self.STATUS_TTL)

    def get_task_status(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        key = f"task_status:{job_id}"
        data = self.client.hgetall(key)
        return data if data else None

    # === 死信队列 ===

    def add_to_dlq(self, task: VideoTask, error: str):
        """将失败任务加入死信队列"""
        data = task.to_dict()
        data["error"] = error
        data["failed_at"] = str(time.time())
        self.client.xadd(self.STREAM_DLQ, data, maxlen=1000)

    # === 向量缓存 ===

    def cache_embeddings(
        self,
        video_id: str,
        embeddings: List[List[float]],
        ttl: int = 3600
    ):
        """缓存向量数据"""
        key = f"embeddings:{video_id}"
        self.client.set(key, json.dumps(embeddings), ex=ttl)

    def get_cached_embeddings(self, video_id: str) -> Optional[List[List[float]]]:
        """获取缓存的向量"""
        key = f"embeddings:{video_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    # === 健康检查 ===

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            self.client.ping()
            info = self.client.info("server")
            video_len = self.client.xlen(self.STREAM_VIDEO_TASKS)
            result_len = self.client.xlen(self.STREAM_RESULT_TASKS)
            return {
                "status": "healthy",
                "redis_version": info.get("redis_version"),
                "video_tasks_len": video_len,
                "result_tasks_len": result_len
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
