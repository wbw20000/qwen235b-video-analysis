"""
Redis 缓存客户端 - 用于 PreProcessor 预处理结果缓存

数据结构:
- Key: preprocessed:{job_id}
- TTL: 300 秒 (5 分钟)
- Value: JSON 格式的预处理结果
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameCache:
    """单帧缓存数据"""
    idx: int
    timestamp_sec: float
    path: str
    embedding: List[float]  # 768 维向量

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "FrameCache":
        return cls(
            idx=int(data["idx"]),
            timestamp_sec=float(data["timestamp_sec"]),
            path=data["path"],
            embedding=data["embedding"]
        )


@dataclass
class PreprocessedResult:
    """预处理结果缓存"""
    job_id: str
    camera_id: str
    video_path: str
    created_at: float
    frames: List[FrameCache]
    trace_id: str = ""

    def to_dict(self) -> Dict:
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "video_path": self.video_path,
            "created_at": self.created_at,
            "trace_id": self.trace_id,
            "frames": [f.to_dict() for f in self.frames]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PreprocessedResult":
        return cls(
            job_id=data["job_id"],
            camera_id=data["camera_id"],
            video_path=data["video_path"],
            created_at=float(data["created_at"]),
            trace_id=data.get("trace_id", ""),
            frames=[FrameCache.from_dict(f) for f in data["frames"]]
        )

    def get_embeddings_array(self) -> np.ndarray:
        """获取所有帧的 embedding 矩阵 (N x 768)"""
        if not self.frames:
            return np.array([])
        return np.array([f.embedding for f in self.frames], dtype=np.float32)


@dataclass
class DownstreamTask:
    """下游分析器任务"""
    job_id: str
    camera_id: str
    cache_key: str
    trace_id: str
    video_path: str = ""
    created_at: float = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "DownstreamTask":
        return cls(
            job_id=data["job_id"],
            camera_id=data["camera_id"],
            cache_key=data["cache_key"],
            trace_id=data["trace_id"],
            video_path=data.get("video_path", ""),
            created_at=float(data.get("created_at", time.time()))
        )


class CacheClient:
    """Redis 缓存客户端"""

    DEFAULT_TTL = 300  # 5 分钟
    KEY_PREFIX = "preprocessed"

    # 下游任务队列
    STREAM_ACCIDENT_TASKS = "accident_tasks"
    STREAM_MV_TASKS = "mv_tasks"
    STREAM_EBIKE_TASKS = "ebike_tasks"
    STREAM_ADS_TASKS = "ads_tasks"

    ALL_DOWNSTREAM_STREAMS = [
        STREAM_ACCIDENT_TASKS,
        STREAM_MV_TASKS,
        STREAM_EBIKE_TASKS,
        STREAM_ADS_TASKS
    ]

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

    def _make_key(self, job_id: str) -> str:
        """生成缓存 key"""
        return f"{self.KEY_PREFIX}:{job_id}"

    def set(
        self,
        job_id: str,
        result: PreprocessedResult,
        ttl: int = None
    ) -> bool:
        """
        存储预处理结果

        Args:
            job_id: 任务 ID
            result: 预处理结果
            ttl: 过期时间（秒），默认 300 秒

        Returns:
            是否成功
        """
        ttl = ttl or self.DEFAULT_TTL
        key = self._make_key(job_id)
        try:
            data = json.dumps(result.to_dict())
            self.client.set(key, data, ex=ttl)
            logger.debug(f"缓存写入成功: {key}, TTL={ttl}s, frames={len(result.frames)}")
            return True
        except Exception as e:
            logger.error(f"缓存写入失败: {key}, error={e}")
            return False

    def get(self, job_id: str) -> Optional[PreprocessedResult]:
        """
        获取预处理结果

        Args:
            job_id: 任务 ID

        Returns:
            预处理结果，不存在返回 None
        """
        key = self._make_key(job_id)
        try:
            data = self.client.get(key)
            if data is None:
                logger.warning(f"缓存未命中: {key}")
                return None
            result = PreprocessedResult.from_dict(json.loads(data))
            logger.debug(f"缓存命中: {key}, frames={len(result.frames)}")
            return result
        except Exception as e:
            logger.error(f"缓存读取失败: {key}, error={e}")
            return None

    def exists(self, job_id: str) -> bool:
        """检查缓存是否存在"""
        key = self._make_key(job_id)
        return self.client.exists(key) > 0

    def delete(self, job_id: str) -> bool:
        """删除缓存"""
        key = self._make_key(job_id)
        return self.client.delete(key) > 0

    def extend_ttl(self, job_id: str, ttl: int = None) -> bool:
        """延长缓存 TTL"""
        ttl = ttl or self.DEFAULT_TTL
        key = self._make_key(job_id)
        return self.client.expire(key, ttl)

    # === 下游任务队列操作 ===

    def ensure_consumer_group(self, stream: str, group: str):
        """确保 Consumer Group 存在"""
        try:
            self.client.xgroup_create(stream, group, id="0", mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def publish_downstream_task(
        self,
        task: DownstreamTask,
        streams: List[str] = None
    ) -> Dict[str, str]:
        """
        发布下游任务到指定队列

        Args:
            task: 下游任务
            streams: 目标队列列表，默认发布到所有 4 个队列

        Returns:
            {stream_name: msg_id} 映射
        """
        streams = streams or self.ALL_DOWNSTREAM_STREAMS
        results = {}
        task_data = task.to_dict()
        for stream in streams:
            try:
                msg_id = self.client.xadd(stream, task_data)
                results[stream] = msg_id
                logger.debug(f"发布下游任务: {stream} -> {msg_id}")
            except Exception as e:
                logger.error(f"发布下游任务失败: {stream}, error={e}")
                results[stream] = None
        return results

    def read_downstream_task(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        block: int = 5000
    ) -> List[tuple]:
        """
        读取下游任务

        Args:
            stream: 队列名称 (accident_tasks/mv_tasks/ebike_tasks/ads_tasks)
            group: Consumer Group 名称
            consumer: Consumer 名称
            count: 一次读取数量
            block: 阻塞等待时间（毫秒）

        Returns:
            [(msg_id, DownstreamTask), ...]
        """
        self.ensure_consumer_group(stream, group)

        # 先读取 pending 消息
        result = self.client.xreadgroup(
            group,
            consumer,
            {stream: "0"},
            count=count,
            block=0
        )

        # 如果没有 pending，读新消息
        if not result or not result[0][1]:
            result = self.client.xreadgroup(
                group,
                consumer,
                {stream: ">"},
                count=count,
                block=block
            )

        if not result:
            return []

        tasks = []
        for stream_name, messages in result:
            for msg_id, data in messages:
                tasks.append((msg_id, DownstreamTask.from_dict(data)))
        return tasks

    def ack_downstream_task(self, stream: str, group: str, msg_id: str):
        """确认下游任务处理完成"""
        self.client.xack(stream, group, msg_id)

    def delete_consumer(self, stream: str, group: str, consumer: str) -> int:
        """删除消费者（优雅退出时清理）

        Args:
            stream: Stream 名称
            group: Consumer Group 名称
            consumer: Consumer 名称

        Returns:
            删除的 pending 消息数量
        """
        try:
            return self.client.xgroup_delconsumer(stream, group, consumer)
        except Exception as e:
            logger.warning(f"删除消费者失败: {consumer}, error={e}")
            return 0

    # === 统计信息 ===

    def get_queue_lengths(self) -> Dict[str, int]:
        """获取所有下游队列长度"""
        return {
            stream: self.client.xlen(stream)
            for stream in self.ALL_DOWNSTREAM_STREAMS
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        keys = list(self.client.scan_iter(f"{self.KEY_PREFIX}:*", count=1000))
        return {
            "cached_jobs": len(keys),
            "queue_lengths": self.get_queue_lengths()
        }
