#!/usr/bin/env python3
"""
多语义分析器回归测试脚本

将 /data1/testdata/multi_semantic 中的视频发送到 Redis Streams，
通过真实的分析器管道进行回归测试。

用法:
    python tools/run_regression_multi_semantic.py --analysis-type mv_violation --limit 10
    python tools/run_regression_multi_semantic.py --all --limit 50
"""
import os
import sys
import time
import json
import uuid
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import redis


@dataclass
class TestResult:
    """测试结果"""
    job_id: str
    video_path: str
    analysis_type: str
    status: str  # pending, processing, done, failed, timeout
    result_path: Optional[str] = None
    judgment: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None


def get_redis_client(host: str = "localhost", port: int = 6379) -> redis.Redis:
    """获取 Redis 客户端"""
    return redis.Redis(host=host, port=port, decode_responses=True)


def scan_videos(base_dir: str, analysis_type: Optional[str] = None) -> List[Dict]:
    """扫描视频文件"""
    videos = []
    base_path = Path(base_dir)

    if analysis_type:
        # 扫描指定类型
        type_dir = base_path / analysis_type
        if type_dir.exists():
            for video_file in type_dir.rglob("*.mp4"):
                videos.append({
                    "path": str(video_file),
                    "analysis_type": analysis_type,
                    "name": video_file.name
                })
    else:
        # 扫描所有类型
        for type_dir in base_path.iterdir():
            if type_dir.is_dir():
                atype = type_dir.name
                for video_file in type_dir.rglob("*.mp4"):
                    videos.append({
                        "path": str(video_file),
                        "analysis_type": atype,
                        "name": video_file.name
                    })

    return videos


def submit_video_task(
    r: redis.Redis,
    video_path: str,
    analysis_type: str,
    camera_id: str = "test_camera"
) -> str:
    """提交视频任务到 Redis Stream"""
    job_id = f"test_{uuid.uuid4().hex[:12]}"
    trace_id = f"trace_{uuid.uuid4().hex[:8]}"

    task = {
        "job_id": job_id,
        "camera_id": camera_id,
        "window_path": video_path,
        "trace_id": trace_id,
        "seg_end_ts": str(int(time.time())),
        "created_at": str(time.time()),
        "analysis_type": analysis_type
    }

    # 发送到 video_tasks stream
    r.xadd("video_tasks", task)

    return job_id


def check_task_status(r: redis.Redis, job_id: str) -> Optional[str]:
    """检查任务状态"""
    status = r.hget("task_status", job_id)
    return status


def wait_for_results(
    r: redis.Redis,
    job_ids: List[str],
    timeout: int = 300,
    poll_interval: int = 5
) -> Dict[str, TestResult]:
    """等待所有任务完成"""
    results = {}
    start_time = time.time()
    pending = set(job_ids)

    while pending and (time.time() - start_time) < timeout:
        for job_id in list(pending):
            status = check_task_status(r, job_id)
            if status in ("done", "failed"):
                pending.remove(job_id)
                results[job_id] = status

        if pending:
            print(f"[{int(time.time() - start_time)}s] 等待中: {len(pending)}/{len(job_ids)}")
            time.sleep(poll_interval)

    # 标记超时
    for job_id in pending:
        results[job_id] = "timeout"

    return results


def main():
    parser = argparse.ArgumentParser(description="多语义分析器回归测试")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--data-dir", default="/data1/testdata/multi_semantic")
    parser.add_argument("--analysis-type", choices=["mv_violation", "ebike_violation", "ads_behavior"],
                        help="指定分析类型，不指定则测试所有类型")
    parser.add_argument("--all", action="store_true", help="测试所有类型")
    parser.add_argument("--limit", type=int, default=10, help="每种类型最多测试的视频数")
    parser.add_argument("--timeout", type=int, default=300, help="等待超时时间(秒)")
    parser.add_argument("--dry-run", action="store_true", help="仅列出视频，不实际提交")
    args = parser.parse_args()

    print("=" * 60)
    print("多语义分析器回归测试")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"Redis: {args.redis_host}:{args.redis_port}")
    print(f"分析类型: {args.analysis_type or '全部'}")
    print(f"每类限制: {args.limit}")
    print()

    # 扫描视频
    analysis_type = None if args.all else args.analysis_type
    videos = scan_videos(args.data_dir, analysis_type)

    if not videos:
        print("未找到视频文件!")
        return

    # 按类型分组
    by_type = {}
    for v in videos:
        atype = v["analysis_type"]
        if atype not in by_type:
            by_type[atype] = []
        by_type[atype].append(v)

    print(f"找到视频文件:")
    for atype, vlist in by_type.items():
        print(f"  {atype}: {len(vlist)} 个")
    print()

    if args.dry_run:
        print("Dry run 模式，不提交任务")
        for atype, vlist in by_type.items():
            print(f"\n[{atype}] 前 {min(args.limit, len(vlist))} 个:")
            for v in vlist[:args.limit]:
                print(f"  - {v['name']}")
        return

    # 连接 Redis
    r = get_redis_client(args.redis_host, args.redis_port)

    try:
        r.ping()
        print("Redis 连接成功")
    except redis.ConnectionError as e:
        print(f"Redis 连接失败: {e}")
        return

    # 提交任务
    submitted = []
    for atype, vlist in by_type.items():
        count = 0
        for v in vlist[:args.limit]:
            job_id = submit_video_task(r, v["path"], atype)
            submitted.append({
                "job_id": job_id,
                "video": v["name"],
                "analysis_type": atype
            })
            count += 1
            print(f"提交: [{atype}] {v['name']} -> {job_id}")
        print(f"[{atype}] 提交 {count} 个任务")

    print()
    print(f"共提交 {len(submitted)} 个任务，等待完成...")
    print()

    # 等待结果
    job_ids = [s["job_id"] for s in submitted]
    results = wait_for_results(r, job_ids, timeout=args.timeout)

    # 统计结果
    print()
    print("=" * 60)
    print("测试结果")
    print("=" * 60)

    stats = {"done": 0, "failed": 0, "timeout": 0}
    for s in submitted:
        status = results.get(s["job_id"], "unknown")
        stats[status] = stats.get(status, 0) + 1
        print(f"  [{s['analysis_type']}] {s['video']}: {status}")

    print()
    print(f"完成: {stats.get('done', 0)}")
    print(f"失败: {stats.get('failed', 0)}")
    print(f"超时: {stats.get('timeout', 0)}")


if __name__ == "__main__":
    main()
