from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

from .config import DataStoreConfig


class DataLoggerAndIndexer:
    """将结果落盘到 SQLite，便于检索/评估。"""

    def __init__(self, config: DataStoreConfig):
        self.config = config
        os.makedirs(os.path.dirname(self.config.sqlite_path), exist_ok=True)
        self._ensure_table()

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

    def log_event(self, record: Dict[str, Any]) -> int:
        conn = sqlite3.connect(self.config.sqlite_path)
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
        conn.close()
        return event_id
