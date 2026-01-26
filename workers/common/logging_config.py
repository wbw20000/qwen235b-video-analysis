"""
统一日志格式配置
所有服务使用 JSON 格式日志，包含 trace_id/job_id/camera_id
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional


class JSONFormatter(logging.Formatter):
    """JSON 格式化器，支持 trace_id 等链路追踪字段"""

    def __init__(self, service_name: str = "unknown"):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "trace_id": getattr(record, "trace_id", ""),
            "job_id": getattr(record, "job_id", ""),
            "camera_id": getattr(record, "camera_id", ""),
            "stage": getattr(record, "stage", ""),
            "msg": record.getMessage(),
            "file": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(
    service_name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置统一格式的日志记录器

    Args:
        service_name: 服务名称
        level: 日志级别
        log_file: 可选的日志文件路径

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(level)

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter(service_name))
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter(service_name))
        logger.addHandler(file_handler)

    return logger


class LogContext:
    """日志上下文管理器，自动附加 trace_id 等字段"""

    def __init__(
        self,
        logger: logging.Logger,
        trace_id: str = "",
        job_id: str = "",
        camera_id: str = "",
        stage: str = ""
    ):
        self.logger = logger
        self.extra = {
            "trace_id": trace_id,
            "job_id": job_id,
            "camera_id": camera_id,
            "stage": stage
        }

    def info(self, msg: str, **kwargs):
        self.logger.info(msg, extra={**self.extra, **kwargs})

    def warning(self, msg: str, **kwargs):
        self.logger.warning(msg, extra={**self.extra, **kwargs})

    def error(self, msg: str, **kwargs):
        self.logger.error(msg, extra={**self.extra, **kwargs})

    def debug(self, msg: str, **kwargs):
        self.logger.debug(msg, extra={**self.extra, **kwargs})
