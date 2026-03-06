#!/usr/bin/env python3
"""
API Gateway - REST API 入口
- 生成 trace_id
- 提供健康检查
- 查询任务状态
- 手动提交视频分析任务
"""
import os
import sys
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, g
from functools import wraps

from workers.common.logging_config import setup_logger, LogContext
from workers.common.redis_client import RedisStreamClient, VideoTask, EdgeEvent

# 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")
AGGREGATED_DIR = os.getenv("AGGREGATED_DIR", "/data1/aggregated")

app = Flask(__name__)
logger = setup_logger("api-gateway")
log = LogContext(logger, stage="api-gateway")

# Redis 客户端
redis_client: Optional[RedisStreamClient] = None


def get_redis() -> RedisStreamClient:
    """获取 Redis 客户端"""
    global redis_client
    if redis_client is None:
        redis_client = RedisStreamClient(REDIS_HOST, REDIS_PORT)
    return redis_client


def generate_trace_id() -> str:
    """生成 trace_id"""
    return str(uuid.uuid4())


def with_trace_id(f):
    """装饰器：添加 trace_id"""
    @wraps(f)
    def decorated(*args, **kwargs):
        # 从 header 或生成新的 trace_id
        trace_id = request.headers.get("X-Trace-Id", generate_trace_id())
        g.trace_id = trace_id
        g.start_time = time.time()

        response = f(*args, **kwargs)

        # 添加 trace_id 到响应头
        if hasattr(response, "headers"):
            response.headers["X-Trace-Id"] = trace_id
            response.headers["X-Request-Time-Ms"] = str(int((time.time() - g.start_time) * 1000))

        return response
    return decorated


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    try:
        redis = get_redis()
        # 检查 Redis 连接
        redis.client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {e}"

    return jsonify({
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "redis": redis_status
        }
    })


@app.route("/api/v1/tasks", methods=["POST"])
@with_trace_id
def submit_task():
    """
    提交视频分析任务

    请求体:
    {
        "video_path": "/path/to/video.mp4",
        "camera_id": "cam_001"
    }

    响应:
    {
        "job_id": "cam_001_1234567890",
        "trace_id": "uuid",
        "status": "queued"
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    video_path = data.get("video_path")
    camera_id = data.get("camera_id", "unknown")

    if not video_path:
        return jsonify({"error": "Missing 'video_path' field"}), 400

    # 检查文件是否存在
    if not Path(video_path).exists():
        return jsonify({"error": f"Video file not found: {video_path}"}), 404

    # 生成 job_id
    seg_end_ts = int(time.time())
    job_id = f"{camera_id}_{seg_end_ts}"
    trace_id = g.trace_id

    # 创建任务
    task = VideoTask(
        job_id=job_id,
        camera_id=camera_id,
        window_path=video_path,
        seg_end_ts=seg_end_ts,
        trace_id=trace_id
    )

    try:
        redis = get_redis()
        redis.add_video_task(task)
        redis.set_task_status(job_id, "queued")

        log.info(
            f"任务已提交: {job_id}",
            job_id=job_id,
            trace_id=trace_id
        )

        return jsonify({
            "job_id": job_id,
            "trace_id": trace_id,
            "status": "queued",
            "video_path": video_path
        })

    except Exception as e:
        log.error(f"提交任务失败: {e}", trace_id=trace_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tasks/<job_id>", methods=["GET"])
@with_trace_id
def get_task_status(job_id: str):
    """
    查询任务状态

    响应:
    {
        "job_id": "cam_001_1234567890",
        "status": "done",
        "worker_id": "analyzer_12345",
        "start_ts": 1234567890,
        "end_ts": 1234567900
    }
    """
    try:
        redis = get_redis()
        status = redis.get_task_status(job_id)

        if not status:
            return jsonify({"error": "Task not found"}), 404

        return jsonify({
            "job_id": job_id,
            **status
        })

    except Exception as e:
        log.error(f"查询任务状态失败: {e}", job_id=job_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/tasks/<job_id>/result", methods=["GET"])
@with_trace_id
def get_task_result(job_id: str):
    """
    获取任务结果

    响应:
    {
        "job_id": "cam_001_1234567890",
        "result": {...}
    }
    """
    import gzip
    import json

    try:
        # 解析 job_id 获取 camera_id
        parts = job_id.rsplit("_", 1)
        if len(parts) != 2:
            return jsonify({"error": "Invalid job_id format"}), 400

        camera_id = parts[0]

        # 查找结果文件
        result_path = Path(RESULTS_DIR) / camera_id / f"{job_id}.result.json.gz"

        if not result_path.exists():
            # 尝试不带 .gz 的路径
            result_path = Path(RESULTS_DIR) / camera_id / f"{job_id}.result.json"

        if not result_path.exists():
            return jsonify({"error": "Result not found"}), 404

        # 加载结果
        if result_path.suffix == ".gz" or ".gz" in result_path.name:
            with gzip.open(result_path, "rt", encoding="utf-8") as f:
                result = json.load(f)
        else:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)

        return jsonify({
            "job_id": job_id,
            "result": result
        })

    except Exception as e:
        log.error(f"获取任务结果失败: {e}", job_id=job_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/cameras/<camera_id>/events", methods=["GET"])
@with_trace_id
def get_camera_events(camera_id: str):
    """
    获取摄像头的聚合事件列表

    查询参数:
    - start_ts: 开始时间戳（可选）
    - end_ts: 结束时间戳（可选）
    - limit: 返回数量限制（默认 100）

    响应:
    {
        "camera_id": "cam_001",
        "events": [...]
    }
    """
    import json

    try:
        start_ts = request.args.get("start_ts", type=int, default=0)
        end_ts = request.args.get("end_ts", type=int, default=int(time.time()))
        limit = request.args.get("limit", type=int, default=100)

        # 查找聚合事件
        events_dir = Path(AGGREGATED_DIR) / camera_id
        if not events_dir.exists():
            return jsonify({
                "camera_id": camera_id,
                "events": [],
                "total": 0
            })

        events = []
        for event_file in events_dir.glob("*.event.json"):
            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    event = json.load(f)

                event_time = event.get("event_time", 0)
                if start_ts <= event_time <= end_ts:
                    events.append(event)
            except Exception as e:
                log.warning(f"加载事件失败: {event_file}, {e}")

        # 按时间排序
        events.sort(key=lambda x: x.get("event_time", 0), reverse=True)
        events = events[:limit]

        return jsonify({
            "camera_id": camera_id,
            "events": events,
            "total": len(events)
        })

    except Exception as e:
        log.error(f"获取摄像头事件失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/edge/events", methods=["POST"])
@with_trace_id
def receive_edge_event():
    """
    接收边缘触发事件

    请求体 (来自 Edge Trigger):
    {
        "camera_id": "cam_001",
        "timestamp": 1234567890,
        "similarity_score": 0.15,
        "keyframe_count": 5,
        "keyframes_base64": ["base64_img1", ...],
        "window_path": "/edge/videos/windows/cam_001/window_123.mp4",
        "trigger_time": "2025-01-31T12:00:00Z"
    }

    响应:
    {
        "status": "accepted",
        "job_id": "cam_001_1234567890",
        "trace_id": "uuid"
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    required_fields = ["camera_id", "timestamp", "similarity_score"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    camera_id = data["camera_id"]
    timestamp = int(data["timestamp"])
    trace_id = g.trace_id
    job_id = f"{camera_id}_{timestamp}"

    try:
        # 创建边缘事件
        event = EdgeEvent(
            camera_id=camera_id,
            timestamp=timestamp,
            similarity_score=float(data["similarity_score"]),
            keyframe_count=int(data.get("keyframe_count", 0)),
            keyframes_base64=data.get("keyframes_base64", []),
            window_path=data.get("window_path", ""),
            trigger_time=data.get("trigger_time", datetime.utcnow().isoformat() + "Z"),
            trace_id=trace_id
        )

        # 添加到边缘事件队列
        redis = get_redis()
        redis.add_edge_event(event)

        log.info(
            f"边缘事件已接收: {job_id}, score={event.similarity_score:.4f}",
            job_id=job_id,
            trace_id=trace_id,
            camera_id=camera_id
        )

        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "trace_id": trace_id,
            "similarity_score": event.similarity_score,
            "keyframe_count": event.keyframe_count
        })

    except Exception as e:
        log.error(f"接收边缘事件失败: {e}", trace_id=trace_id)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/edge/events", methods=["GET"])
@with_trace_id
def list_edge_events():
    """
    查询边缘事件队列状态

    查询参数:
    - limit: 返回数量限制（默认 10）

    响应:
    {
        "queue_length": 5,
        "events": [...]
    }
    """
    try:
        limit = request.args.get("limit", type=int, default=10)
        redis = get_redis()

        # 获取队列长度
        queue_len = redis.client.xlen(RedisStreamClient.STREAM_EDGE_EVENTS)

        # 读取最近的事件（不消费，仅查看）
        events = redis.client.xrange(
            RedisStreamClient.STREAM_EDGE_EVENTS,
            "-", "+",
            count=limit
        )

        event_list = []
        for msg_id, data in events:
            event_list.append({
                "msg_id": msg_id,
                "camera_id": data.get("camera_id"),
                "timestamp": data.get("timestamp"),
                "similarity_score": data.get("similarity_score"),
                "trigger_time": data.get("trigger_time")
            })

        return jsonify({
            "queue_length": queue_len,
            "events": event_list
        })

    except Exception as e:
        log.error(f"查询边缘事件失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/stats", methods=["GET"])
@with_trace_id
def get_stats():
    """
    获取系统统计信息

    响应:
    {
        "queues": {
            "video_tasks": {"length": 10, "pending": 5},
            "result_tasks": {"length": 3, "pending": 1}
        },
        "cameras": ["cam_001", "cam_002"],
        "timestamp": "2025-01-26T12:00:00Z"
    }
    """
    try:
        redis = get_redis()

        # 队列统计
        video_len = redis.client.xlen(RedisStreamClient.STREAM_VIDEO_TASKS)
        result_len = redis.client.xlen(RedisStreamClient.STREAM_RESULT_TASKS)
        edge_len = redis.client.xlen(RedisStreamClient.STREAM_EDGE_EVENTS)

        # 摄像头列表
        cameras = []
        results_path = Path(RESULTS_DIR)
        if results_path.exists():
            cameras = [d.name for d in results_path.iterdir() if d.is_dir()]

        return jsonify({
            "queues": {
                "video_tasks": {"length": video_len},
                "result_tasks": {"length": result_len},
                "edge_events": {"length": edge_len}
            },
            "cameras": cameras,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

    except Exception as e:
        log.error(f"获取统计信息失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


def main():
    import argparse

    parser = argparse.ArgumentParser(description="API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=5000, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    args = parser.parse_args()

    log.info(f"启动 API Gateway: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
