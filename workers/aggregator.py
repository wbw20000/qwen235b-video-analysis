#!/usr/bin/env python3
"""
Aggregator Worker - 结果聚合 + 去重
- 消费 result_tasks 队列
- 按 (camera_id, event_time ± 15s) 去重
- 保留 VLM confidence 最高的结果
- 幂等落盘（原子写入）
- 更新 task_status
"""
import os
import sys
import time
import signal
import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, ResultTask

# 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")
AGGREGATED_DIR = os.getenv("AGGREGATED_DIR", "/data1/aggregated")
DEDUP_WINDOW_SEC = int(os.getenv("DEDUP_WINDOW_SEC", "15"))  # ±15秒去重窗口
CONSUMER_GROUP = "aggregators"
CONSUMER_NAME = os.getenv("HOSTNAME", f"aggregator_{os.getpid()}")
MAX_TASKS_BEFORE_EXIT = int(os.getenv("MAX_TASKS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # 批量处理数量

logger = setup_logger("aggregator")
log = LogContext(logger, stage="aggregator")


@dataclass
class AggregatedEvent:
    """聚合后的事件"""
    event_id: str
    camera_id: str
    event_time: int  # 事件时间戳（取最高置信度结果的时间）
    is_accident: bool
    confidence: float
    result_count: int  # 原始结果数量
    result_paths: List[str]  # 原始结果文件路径
    best_result_path: str  # 最高置信度结果路径
    created_at: str


class Aggregator:
    """结果聚合器"""

    def __init__(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        results_dir: str = RESULTS_DIR,
        aggregated_dir: str = AGGREGATED_DIR,
        dedup_window_sec: int = DEDUP_WINDOW_SEC
    ):
        self.redis = RedisStreamClient(redis_host, redis_port)
        self.results_dir = Path(results_dir)
        self.aggregated_dir = Path(aggregated_dir)
        self.aggregated_dir.mkdir(parents=True, exist_ok=True)
        self.dedup_window_sec = dedup_window_sec

        self.task_count = 0
        self.running = True

        # 内存中的事件缓存（用于去重）
        # key: camera_id, value: list of (event_time, confidence, result_path)
        self.event_cache: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)

        # 信号处理
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        log.info(f"收到信号 {signum}，准备优雅退出")
        self.running = False

    def _load_result(self, result_path: str) -> Optional[Dict]:
        """加载结果文件"""
        try:
            path = Path(result_path)
            if not path.exists():
                return None

            if path.suffix == ".gz" or ".gz" in path.name:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            log.error(f"加载结果失败: {result_path}, {e}")
            return None

    def _is_duplicate(
        self,
        camera_id: str,
        event_time: int,
        confidence: float
    ) -> Tuple[bool, Optional[int]]:
        """
        检查是否为重复事件
        返回: (是否重复, 如果重复且新事件置信度更高则返回要替换的索引)
        """
        events = self.event_cache[camera_id]

        for i, (cached_time, cached_conf, cached_path) in enumerate(events):
            time_diff = abs(event_time - cached_time)
            if time_diff <= self.dedup_window_sec:
                # 在去重窗口内
                if confidence > cached_conf:
                    # 新事件置信度更高，返回替换索引
                    return True, i
                else:
                    # 旧事件置信度更高，直接去重
                    return True, None

        return False, None

    def _add_event(
        self,
        camera_id: str,
        event_time: int,
        confidence: float,
        result_path: str,
        replace_idx: Optional[int] = None
    ):
        """添加事件到缓存"""
        if replace_idx is not None:
            # 替换旧事件
            self.event_cache[camera_id][replace_idx] = (event_time, confidence, result_path)
            log.info(f"替换事件: camera={camera_id}, time={event_time}, conf={confidence}")
        else:
            # 添加新事件
            self.event_cache[camera_id].append((event_time, confidence, result_path))
            log.info(f"添加事件: camera={camera_id}, time={event_time}, conf={confidence}")

        # 清理过期事件（超过 1 小时的事件）
        self._cleanup_old_events(camera_id)

    def _cleanup_old_events(self, camera_id: str, max_age_sec: int = 3600):
        """清理过期事件"""
        current_time = int(time.time())
        events = self.event_cache[camera_id]
        self.event_cache[camera_id] = [
            (t, c, p) for t, c, p in events
            if current_time - t < max_age_sec
        ]

    def _save_aggregated_event(
        self,
        camera_id: str,
        event_time: int,
        is_accident: bool,
        confidence: float,
        result_path: str
    ) -> Optional[Path]:
        """
        保存聚合后的事件（原子写入）
        """
        event_id = f"{camera_id}_{event_time}"

        event = AggregatedEvent(
            event_id=event_id,
            camera_id=camera_id,
            event_time=event_time,
            is_accident=is_accident,
            confidence=confidence,
            result_count=1,
            result_paths=[result_path],
            best_result_path=result_path,
            created_at=datetime.utcnow().isoformat() + "Z"
        )

        # 输出路径
        output_dir = self.aggregated_dir / camera_id
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{event_id}.event.json"
        tmp_path = output_dir / f"{filename}.tmp"
        final_path = output_dir / filename

        # 原子写入
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(asdict(event), f, ensure_ascii=False, indent=2)

            # fsync + rename
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
            os.rename(tmp_path, final_path)

            log.info(f"事件已保存: {final_path}")
            return final_path

        except Exception as e:
            log.error(f"保存事件失败: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            return None

    def _generate_daily_summary(self, camera_id: str, date: str) -> Optional[Path]:
        """
        生成每日汇总报告
        """
        events = self.event_cache.get(camera_id, [])
        if not events:
            return None

        # 筛选当日事件
        # date 格式: YYYYMMDD
        try:
            from datetime import datetime as dt
            target_date = dt.strptime(date, "%Y%m%d").date()
        except ValueError:
            return None

        daily_events = []
        for event_time, confidence, result_path in events:
            event_date = datetime.fromtimestamp(event_time).date()
            if event_date == target_date:
                daily_events.append({
                    "event_time": event_time,
                    "confidence": confidence,
                    "result_path": result_path
                })

        if not daily_events:
            return None

        summary = {
            "camera_id": camera_id,
            "date": date,
            "total_events": len(daily_events),
            "events": daily_events,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

        # 输出路径
        output_dir = self.aggregated_dir / camera_id / "daily"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"summary_{date}.json"
        final_path = output_dir / filename

        try:
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            log.info(f"每日汇总已生成: {final_path}")
            return final_path
        except Exception as e:
            log.error(f"生成汇总失败: {e}")
            return None

    def process_result(self, task: ResultTask) -> bool:
        """处理单个结果任务"""
        job_id = task.job_id
        trace_id = task.trace_id

        log.info(
            f"处理结果: {task.result_path}",
            job_id=job_id,
            trace_id=trace_id
        )

        try:
            # 加载结果
            result_data = self._load_result(task.result_path)
            if not result_data:
                log.warning(f"结果文件为空或不存在: {task.result_path}", job_id=job_id)
                return True  # 标记为处理完成，跳过

            # 提取事件信息
            is_accident = task.is_accident
            confidence = task.confidence
            event_time = task.seg_end_ts
            camera_id = task.camera_id

            # 只处理事故事件
            if not is_accident:
                log.debug(f"非事故事件，跳过聚合: {job_id}", job_id=job_id)
                return True

            # 去重检查
            is_dup, replace_idx = self._is_duplicate(camera_id, event_time, confidence)

            if is_dup and replace_idx is None:
                # 重复且置信度不更高，跳过
                log.info(
                    f"重复事件已去重: camera={camera_id}, time={event_time}",
                    job_id=job_id,
                    trace_id=trace_id
                )
                return True

            # 添加/替换事件
            self._add_event(camera_id, event_time, confidence, task.result_path, replace_idx)

            # 保存聚合事件
            self._save_aggregated_event(
                camera_id=camera_id,
                event_time=event_time,
                is_accident=is_accident,
                confidence=confidence,
                result_path=task.result_path
            )

            log.info(
                f"结果聚合完成: {job_id}",
                job_id=job_id,
                trace_id=trace_id
            )

            return True

        except Exception as e:
            log.error(f"处理结果失败: {e}", job_id=job_id, trace_id=trace_id)
            return False

    def run(self):
        """主循环"""
        log.info(f"启动 Aggregator: {CONSUMER_NAME}")

        while self.running:
            # 检查自愈阈值
            if self.task_count >= MAX_TASKS_BEFORE_EXIT:
                log.info(f"达到自愈阈值 ({MAX_TASKS_BEFORE_EXIT})，准备退出")
                break

            try:
                # 批量读取结果任务
                tasks = self.redis.read_result_tasks(
                    group=CONSUMER_GROUP,
                    consumer=CONSUMER_NAME,
                    count=BATCH_SIZE,
                    block=5000
                )

                for msg_id, task in tasks:
                    if not self.running:
                        break

                    if self.process_result(task):
                        self.redis.ack_result_task(CONSUMER_GROUP, msg_id)
                        self.task_count += 1

                if tasks:
                    log.info(f"已处理 {self.task_count}/{MAX_TASKS_BEFORE_EXIT} 个结果")

            except Exception as e:
                log.error(f"主循环异常: {e}")
                time.sleep(5)

        # 生成退出前的每日汇总
        today = datetime.now().strftime("%Y%m%d")
        for camera_id in self.event_cache.keys():
            self._generate_daily_summary(camera_id, today)

        log.info("Aggregator 退出")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregator Worker")
    parser.add_argument("--redis-host", default=REDIS_HOST, help="Redis 主机")
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT, help="Redis 端口")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="结果目录")
    parser.add_argument("--aggregated-dir", default=AGGREGATED_DIR, help="聚合输出目录")
    parser.add_argument(
        "--dedup-window",
        type=int,
        default=DEDUP_WINDOW_SEC,
        help="去重窗口（秒）"
    )
    args = parser.parse_args()

    aggregator = Aggregator(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        results_dir=args.results_dir,
        aggregated_dir=args.aggregated_dir,
        dedup_window_sec=args.dedup_window
    )
    aggregator.run()


if __name__ == "__main__":
    main()
