#!/usr/bin/env python3
"""
VLM调试日志系统
- 支持环境变量配置开关
- 记录请求/响应详情
- 支持采样率控制

环境变量配置:
  VLM_DEBUG_ENABLED=true/false     # 总开关
  VLM_DEBUG_LEVEL=DEBUG/INFO/WARNING/ERROR  # 日志级别
  VLM_DEBUG_LOG_REQUEST=true/false  # 记录请求
  VLM_DEBUG_LOG_RESPONSE=true/false # 记录响应
  VLM_DEBUG_LOG_IMAGES=true/false   # 记录图片内容(谨慎开启)
  VLM_DEBUG_SAMPLE_RATE=1.0         # 采样率 0.0-1.0
  VLM_DEBUG_LOG_DIR=/tmp/vlm_debug  # 日志目录
  VLM_DEBUG_CONSOLE=true/false      # 控制台输出
"""
import os
import json
import time
import hashlib
import logging
import random
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict

# ========== 配置 ==========
VLM_DEBUG_ENABLED = os.getenv("VLM_DEBUG_ENABLED", "true").lower() == "true"
VLM_DEBUG_LEVEL = os.getenv("VLM_DEBUG_LEVEL", "INFO")
VLM_DEBUG_LOG_REQUEST = os.getenv("VLM_DEBUG_LOG_REQUEST", "true").lower() == "true"
VLM_DEBUG_LOG_RESPONSE = os.getenv("VLM_DEBUG_LOG_RESPONSE", "true").lower() == "true"
VLM_DEBUG_LOG_IMAGES = os.getenv("VLM_DEBUG_LOG_IMAGES", "false").lower() == "true"
VLM_DEBUG_SAMPLE_RATE = float(os.getenv("VLM_DEBUG_SAMPLE_RATE", "1.0"))
VLM_DEBUG_LOG_DIR = os.getenv("VLM_DEBUG_LOG_DIR", "/tmp/vlm_debug")
VLM_DEBUG_MAX_BODY_SIZE = int(os.getenv("VLM_DEBUG_MAX_BODY_SIZE", "10000"))
VLM_DEBUG_CONSOLE = os.getenv("VLM_DEBUG_CONSOLE", "false").lower() == "true"

# 创建日志目录
if VLM_DEBUG_ENABLED:
    os.makedirs(VLM_DEBUG_LOG_DIR, exist_ok=True)

# 配置logger
logger = logging.getLogger("vlm_debug")
logger.setLevel(getattr(logging, VLM_DEBUG_LEVEL, logging.INFO))

if VLM_DEBUG_ENABLED and not logger.handlers:
    # 文件handler - 按日期滚动
    log_file = os.path.join(VLM_DEBUG_LOG_DIR, f"vlm_debug_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)

    # 控制台handler
    if VLM_DEBUG_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s [VLM_DEBUG] %(message)s"
        ))
        logger.addHandler(console_handler)


@dataclass
class RequestInfo:
    """请求信息"""
    job_id: str
    trace_id: str
    url: str
    model: str
    max_tokens: int
    temperature: float
    image_count: int
    total_image_size_kb: float
    content_structure: List[str]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ResponseInfo:
    """响应信息"""
    job_id: str
    trace_id: str
    status_code: int
    elapsed_sec: float
    success: bool
    response_id: str = ""
    model: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error_detail: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


def should_sample() -> bool:
    """根据采样率决定是否记录"""
    return random.random() < VLM_DEBUG_SAMPLE_RATE


def truncate_base64(content: Any, max_preview: int = 100) -> Any:
    """截断base64图片内容，只保留摘要"""
    if isinstance(content, str):
        if len(content) > 1000 and "base64" in content[:50]:
            # 提取图片类型和大小
            if "," in content:
                parts = content.split(",", 1)
                img_type = parts[0]
                img_data = parts[1]
                md5_hash = hashlib.md5(img_data[:1000].encode()).hexdigest()[:8]
                return f"{img_type},[LEN={len(img_data)},MD5={md5_hash}]"
            return f"[BASE64_LEN={len(content)}]"
    elif isinstance(content, dict):
        return {k: truncate_base64(v, max_preview) for k, v in content.items()}
    elif isinstance(content, list):
        return [truncate_base64(item, max_preview) for item in content]
    return content


class VLMDebugLogger:
    """VLM调试日志记录器"""

    def __init__(self, component: str = "unknown"):
        self.component = component
        self.enabled = VLM_DEBUG_ENABLED
        self._request_start_times: Dict[str, float] = {}

    def start_request(self, job_id: str) -> float:
        """标记请求开始时间"""
        start_time = time.time()
        self._request_start_times[job_id] = start_time
        return start_time

    def get_elapsed(self, job_id: str) -> float:
        """获取请求耗时"""
        start_time = self._request_start_times.pop(job_id, None)
        if start_time:
            return time.time() - start_time
        return 0.0

    def log_request(
        self,
        job_id: str,
        trace_id: str,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ):
        """记录VLM请求"""
        if not self.enabled or not VLM_DEBUG_LOG_REQUEST:
            return
        if not should_sample():
            return

        try:
            # 标记开始时间
            self.start_request(job_id)

            # 提取关键信息
            messages = payload.get("messages", [])
            content_info = []
            image_count = 0
            total_image_size = 0

            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        item_type = item.get("type", "unknown")
                        if item_type == "image_url":
                            image_count += 1
                            url_data = item.get("image_url", {}).get("url", "")
                            if url_data.startswith("data:"):
                                b64_part = url_data.split(",", 1)[-1] if "," in url_data else ""
                                total_image_size += len(b64_part)
                            content_info.append(f"image({len(url_data)//1024}KB)")
                        elif item_type == "text":
                            text = item.get("text", "")
                            content_info.append(f"text({len(text)}chars)")
                elif isinstance(content, str):
                    content_info.append(f"str({len(content)})")

            info = RequestInfo(
                job_id=job_id,
                trace_id=trace_id,
                url=url,
                model=payload.get("model", "unknown"),
                max_tokens=payload.get("max_tokens", 0),
                temperature=payload.get("temperature", 0.0),
                image_count=image_count,
                total_image_size_kb=round(total_image_size / 1024, 2),
                content_structure=content_info
            )

            log_entry = {
                "event": "VLM_REQUEST",
                "component": self.component,
                **asdict(info)
            }

            logger.info(json.dumps(log_entry, ensure_ascii=False))

        except Exception as e:
            logger.error(f"[{self.component}] 记录请求失败: {e}")

    def log_response(
        self,
        job_id: str,
        trace_id: str,
        status_code: int,
        response_body: str,
        elapsed_sec: Optional[float] = None,
        error: Optional[str] = None
    ):
        """记录VLM响应"""
        if not self.enabled or not VLM_DEBUG_LOG_RESPONSE:
            return
        if not should_sample():
            return

        try:
            # 获取耗时
            if elapsed_sec is None:
                elapsed_sec = self.get_elapsed(job_id)

            # 初始化响应信息
            info = ResponseInfo(
                job_id=job_id,
                trace_id=trace_id,
                status_code=status_code,
                elapsed_sec=round(elapsed_sec, 2),
                success=(status_code == 200)
            )

            # 解析响应
            if response_body:
                try:
                    resp_json = json.loads(response_body)
                    info.response_id = resp_json.get("id", "")
                    info.model = resp_json.get("model", "")

                    choices = resp_json.get("choices", [])
                    if choices:
                        info.finish_reason = choices[0].get("finish_reason", "")

                    usage = resp_json.get("usage", {})
                    info.prompt_tokens = usage.get("prompt_tokens", 0)
                    info.completion_tokens = usage.get("completion_tokens", 0)

                    # 错误响应记录详情
                    if status_code >= 400:
                        info.error_detail = response_body[:VLM_DEBUG_MAX_BODY_SIZE]

                except json.JSONDecodeError:
                    if status_code >= 400:
                        info.error_detail = response_body[:VLM_DEBUG_MAX_BODY_SIZE]

            if error:
                info.error_detail = str(error)[:VLM_DEBUG_MAX_BODY_SIZE]

            log_entry = {
                "event": "VLM_RESPONSE",
                "component": self.component,
                **asdict(info)
            }

            if status_code >= 400:
                logger.warning(json.dumps(log_entry, ensure_ascii=False))
            else:
                logger.info(json.dumps(log_entry, ensure_ascii=False))

        except Exception as e:
            logger.error(f"[{self.component}] 记录响应失败: {e}")

    def log_error(
        self,
        job_id: str,
        trace_id: str,
        error_type: str,
        error_msg: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录错误详情"""
        if not self.enabled:
            return

        log_entry = {
            "event": "VLM_ERROR",
            "component": self.component,
            "job_id": job_id,
            "trace_id": trace_id,
            "error_type": error_type,
            "error_msg": str(error_msg)[:1000],
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }

        logger.error(json.dumps(log_entry, ensure_ascii=False))

    def log_keyframes(
        self,
        job_id: str,
        trace_id: str,
        keyframes: List[Any],
        context: str = ""
    ):
        """记录关键帧信息"""
        if not self.enabled:
            return

        frame_info = []
        for i, kf in enumerate(keyframes):
            info = {
                "idx": i,
                "path": getattr(kf, "frame_path", str(kf)),
                "exists": os.path.exists(getattr(kf, "frame_path", str(kf))),
            }
            if hasattr(kf, "timestamp_sec"):
                info["ts"] = kf.timestamp_sec
            if hasattr(kf, "similarity_score"):
                info["score"] = kf.similarity_score
            frame_info.append(info)

        log_entry = {
            "event": "VLM_KEYFRAMES",
            "component": self.component,
            "job_id": job_id,
            "trace_id": trace_id,
            "context": context,
            "frame_count": len(keyframes),
            "frames": frame_info,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(json.dumps(log_entry, ensure_ascii=False))


# 全局实例工厂
_loggers: Dict[str, VLMDebugLogger] = {}


def get_logger(component: str) -> VLMDebugLogger:
    """获取指定组件的调试logger"""
    if component not in _loggers:
        _loggers[component] = VLMDebugLogger(component)
    return _loggers[component]


def get_config() -> Dict[str, Any]:
    """获取当前调试配置"""
    return {
        "enabled": VLM_DEBUG_ENABLED,
        "level": VLM_DEBUG_LEVEL,
        "log_request": VLM_DEBUG_LOG_REQUEST,
        "log_response": VLM_DEBUG_LOG_RESPONSE,
        "log_images": VLM_DEBUG_LOG_IMAGES,
        "sample_rate": VLM_DEBUG_SAMPLE_RATE,
        "log_dir": VLM_DEBUG_LOG_DIR,
        "max_body_size": VLM_DEBUG_MAX_BODY_SIZE,
        "console": VLM_DEBUG_CONSOLE,
    }


def set_enabled(enabled: bool):
    """动态设置开关"""
    global VLM_DEBUG_ENABLED
    VLM_DEBUG_ENABLED = enabled
    for lg in _loggers.values():
        lg.enabled = enabled


if __name__ == "__main__":
    print("VLM Debug Logger 配置:")
    print(json.dumps(get_config(), indent=2))
