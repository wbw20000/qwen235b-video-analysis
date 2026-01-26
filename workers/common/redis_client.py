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
    seg_end_ts: int
    trace_id: str
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "VideoTask":
        return cls(
            job_id=data["job_id"],
            camera_id=data["camera_id"],
            window_path=data["window_path"],
            seg_end_ts=int(data["seg_end_ts"]),
            trace_id=data["trace_id"],
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
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, str]:
        d = asdict(self)
        d["is_accident"] = "1" if self.is_accident else "0"
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
            created_at=float(data.get("created_at", time.time()))
        )


class RedisStreamClient:
    """Redis Streams 客户端"""

    STREAM_VIDEO_TASKS = "video_tasks"
    STREAM_RESULT_TASKS = "result_tasks"
    STREAM_DLQ = "dlq_tasks"

    MAX_STREAM_LEN = 10000
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
            maxlen=self.MAX_STREAM_LEN
        )
        # 更新任务状态
        self.set_task_status(task.job_id, "pending")
        return msg_id

    def read_video_tasks(
        self,
        group: str,
        consumer: str,
        count: int = 1,
        block: int = 5000
    ) -> List[tuple]:
        """读取视频任务（阻塞模式）"""
        self.ensure_consumer_group(self.STREAM_VIDEO_TASKS, group)
        result = self.client.xreadgroup(
            group,
            consumer,
            {self.STREAM_VIDEO_TASKS: ">"},
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
            maxlen=self.MAX_STREAM_LEN
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
