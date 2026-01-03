from __future__ import annotations

import json
import os
import sqlite3
import threading
from queue import Queue, Empty
from datetime import datetime
from typing import Any, Dict, Optional


class DataLoggerAndIndexer:
    """
    将结果落盘到 SQLite，便于检索/评估。

    使用单写线程模式避免多线程并发时的 "database is locked" 错误。
    """

    def __init__(self, config: "DataStoreConfig"):
        self.config = config
        os.makedirs(os.path.dirname(self.config.sqlite_path), exist_ok=True)
        self._ensure_table()

        # 单写线程模式
        self._write_queue: Queue = Queue(maxsize=100)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="SQLiteWriter",
            daemon=True
        )
        self._writer_thread.start()
        print("[DataLogger] SQLite单写线程已启动")

    def _ensure_table(self):
        conn = sqlite3.connect(self.config.sqlite_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS traffic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                date TEXT,
                video_path TEXT,
                clip_id TEXT,
                clip_start_time REAL,
                clip_end_time REAL,
                clip_score REAL,
                keyframe_paths TEXT,
                motion_method TEXT,
                query_raw TEXT,
                query_templates TEXT,
                vlm_has_violation INTEGER,
                vlm_violations_json TEXT,
                vlm_text_summary TEXT,
                traffic_light_info TEXT,
                is_true_positive INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def _writer_loop(self):
        """单线程写入循环 - 避免 SQLite 并发写入问题"""
        conn = sqlite3.connect(self.config.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL")  # 启用WAL模式，提高并发性能

        while not self._stop_event.is_set():
            try:
                # 等待写入任务，超时1秒后检查停止标志
                record = self._write_queue.get(timeout=1)
                self._write_event_to_db(conn, record)
                self._write_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[DataLogger] 写入错误: {e}")

        # 清空队列中剩余的任务
        while not self._write_queue.empty():
            try:
                record = self._write_queue.get_nowait()
                self._write_event_to_db(conn, record)
                self._write_queue.task_done()
            except Empty:
                break
            except Exception as e:
                print(f"[DataLogger] 清空队列时写入错误: {e}")

        conn.close()
        print("[DataLogger] SQLite单写线程已停止")

    def _write_event_to_db(self, conn: sqlite3.Connection, record: Dict[str, Any]) -> int:
        """实际写入数据库的方法"""
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute(
            """
            INSERT INTO traffic_events (
                camera_id, date, video_path, clip_id, clip_start_time, clip_end_time, clip_score,
                keyframe_paths, motion_method, query_raw, query_templates, vlm_has_violation,
                vlm_violations_json, vlm_text_summary, traffic_light_info, is_true_positive,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("camera_id"),
                record.get("date"),
                record.get("video_path"),
                record.get("clip_id"),
                record.get("clip_start_time"),
                record.get("clip_end_time"),
                record.get("clip_score"),
                json.dumps(record.get("keyframe_paths", []), ensure_ascii=False),
                record.get("motion_method"),
                record.get("query_raw"),
                json.dumps(record.get("query_templates", []), ensure_ascii=False),
                1 if record.get("vlm_has_violation") else 0,
                json.dumps(record.get("vlm_violations_json"), ensure_ascii=False),
                record.get("vlm_text_summary"),
                json.dumps(record.get("traffic_light_info"), ensure_ascii=False),
                record.get("is_true_positive"),
                now,
                now,
            ),
        )
        event_id = c.lastrowid
        conn.commit()
        return event_id

    def log_event(self, record: Dict[str, Any]) -> None:
        """
        记录事件（非阻塞，放入队列）

        Args:
            record: 事件记录字典
        """
        try:
            self._write_queue.put(record, timeout=5)
        except Exception as e:
            print(f"[DataLogger] 写入队列已满或超时，丢弃事件: {e}")

    def flush(self) -> None:
        """等待队列清空（所有待写入事件完成）"""
        self._write_queue.join()

    def close(self) -> None:
        """关闭写入线程"""
        print("[DataLogger] 正在关闭SQLite单写线程...")
        self.flush()  # 先等待队列清空
        self._stop_event.set()  # 设置停止标志
        self._writer_thread.join(timeout=5)  # 等待线程结束
        print("[DataLogger] SQLite单写线程已关闭")

    def __del__(self):
        """析构函数，确保资源释放"""
        if hasattr(self, '_stop_event') and not self._stop_event.is_set():
            self.close()
