#!/usr/bin/env python3
"""Redis Streams Exporter for Prometheus

监控 Redis Streams 的 backlog 和 Consumer Group 状态
"""
import time
import redis
from prometheus_client import start_http_server, Gauge

# 定义 Prometheus 指标
STREAM_LENGTH = Gauge(
    "redis_stream_length",
    "Number of messages in the stream",
    ["stream"]
)
STREAM_PENDING = Gauge(
    "redis_stream_pending_messages",
    "Number of pending messages in consumer group",
    ["stream", "group"]
)
STREAM_LAG = Gauge(
    "redis_stream_consumer_lag",
    "Lag for each consumer (messages behind)",
    ["stream", "group", "consumer"]
)
STREAM_LAST_DELIVERY_MS = Gauge(
    "redis_stream_last_delivery_ms",
    "Milliseconds since last message delivery",
    ["stream", "group"]
)

# 配置
REDIS_HOST = "redis-0.redis.traffic-vlm.svc.cluster.local"
REDIS_PORT = 6379
STREAMS = ["video_tasks", "analysis_results"]
SCRAPE_INTERVAL = 15


def collect_metrics(r: redis.Redis):
    """收集 Redis Streams 指标"""
    for stream in STREAMS:
        try:
            # Stream 长度
            length = r.xlen(stream)
            STREAM_LENGTH.labels(stream=stream).set(length)
            
            # 获取 Consumer Groups
            try:
                groups = r.xinfo_groups(stream)
                for group in groups:
                    group_name = group["name"]
                    pending = group["pending"]
                    STREAM_PENDING.labels(stream=stream, group=group_name).set(pending)
                    
                    # 最后投递时间
                    last_id = group.get("last-delivered-id", "0-0")
                    if last_id and last_id != "0-0":
                        # 解析时间戳
                        ts_ms = int(last_id.split("-")[0])
                        age_ms = int(time.time() * 1000) - ts_ms
                        STREAM_LAST_DELIVERY_MS.labels(
                            stream=stream, group=group_name
                        ).set(age_ms)
                    
                    # Consumer lag
                    try:
                        consumers = r.xinfo_consumers(stream, group_name)
                        for consumer in consumers:
                            consumer_name = consumer["name"]
                            consumer_pending = consumer["pending"]
                            STREAM_LAG.labels(
                                stream=stream,
                                group=group_name,
                                consumer=consumer_name
                            ).set(consumer_pending)
                    except Exception:
                        pass
                        
            except redis.ResponseError:
                # No consumer groups
                pass
                
        except redis.ResponseError as e:
            # Stream does not exist
            STREAM_LENGTH.labels(stream=stream).set(0)


def main():
    """Main entry point"""
    print(f"Starting Redis Streams Exporter on :9122")
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    print(f"Monitoring streams: {STREAMS}")
    
    # 启动 Prometheus HTTP server
    start_http_server(9122)
    
    # 连接 Redis
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    
    # 循环收集指标
    while True:
        try:
            collect_metrics(r)
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    main()
