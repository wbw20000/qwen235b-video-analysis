#!/usr/bin/env python3
"""
VLM Proxy - 并发阀门
- 最大并发: 2（防止 vLLM 被洪峰打爆）
- 超时: 120s
- 自动重试: 2次
- 统一 trace_id 透传
"""
import os
import sys
import asyncio
import time
from typing import Optional
from dataclasses import dataclass

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from workers.common.logging_config import setup_logger, LogContext

# 配置
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "2.0"))

app = FastAPI(title="VLM Proxy", description="VLM 并发控制代理")
logger = setup_logger("vlm-proxy")
log = LogContext(logger, stage="vlm-proxy")

# 并发信号量
semaphore: Optional[asyncio.Semaphore] = None

# 统计
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "retried_requests": 0,
    "current_concurrent": 0,
    "max_concurrent_seen": 0
}


@app.on_event("startup")
async def startup():
    global semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    log.info(f"VLM Proxy 启动: max_concurrent={MAX_CONCURRENT}, timeout={REQUEST_TIMEOUT}s")


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "vllm_base_url": VLLM_BASE_URL,
        "max_concurrent": MAX_CONCURRENT,
        "stats": stats
    }


@app.get("/stats")
async def get_stats():
    """获取统计信息"""
    return stats


async def forward_request(
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    trace_id: str,
    job_id: str
) -> httpx.Response:
    """
    转发请求到 vLLM，带重试逻辑
    """
    url = f"{VLLM_BASE_URL.rstrip('/')}/{path.lstrip('/')}"

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    stats["retried_requests"] += 1
                    log.warning(
                        f"重试请求 (attempt {attempt + 1}/{MAX_RETRIES + 1})",
                        trace_id=trace_id,
                        job_id=job_id
                    )
                    await asyncio.sleep(RETRY_DELAY)

                response = await client.request(
                    method=method,
                    url=url,
                    headers={k: v for k, v in headers.items() if k.lower() not in ["host", "content-length"]},
                    content=body
                )

                if response.status_code < 500:
                    return response

                last_error = f"vLLM 返回 {response.status_code}"
                log.warning(last_error, trace_id=trace_id, job_id=job_id)

            except httpx.TimeoutException:
                last_error = f"请求超时 ({REQUEST_TIMEOUT}s)"
                log.warning(last_error, trace_id=trace_id, job_id=job_id)

            except httpx.ConnectError as e:
                last_error = f"连接失败: {e}"
                log.warning(last_error, trace_id=trace_id, job_id=job_id)

            except Exception as e:
                last_error = f"请求异常: {e}"
                log.error(last_error, trace_id=trace_id, job_id=job_id)

        # 所有重试失败
        raise HTTPException(status_code=503, detail=f"vLLM 服务不可用: {last_error}")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    """
    代理所有请求到 vLLM
    """
    global stats

    # 提取 trace_id 和 job_id（从 header 或 body）
    trace_id = request.headers.get("X-Trace-Id", "")
    job_id = request.headers.get("X-Job-Id", "")

    stats["total_requests"] += 1

    # 获取信号量（限流）
    async with semaphore:
        stats["current_concurrent"] += 1
        stats["max_concurrent_seen"] = max(
            stats["max_concurrent_seen"],
            stats["current_concurrent"]
        )

        try:
            log.info(
                f"转发请求: {request.method} /{path}",
                trace_id=trace_id,
                job_id=job_id
            )

            body = await request.body()
            headers = dict(request.headers)

            response = await forward_request(
                method=request.method,
                path=path,
                headers=headers,
                body=body,
                trace_id=trace_id,
                job_id=job_id
            )

            stats["successful_requests"] += 1

            return JSONResponse(
                content=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                status_code=response.status_code,
                headers={
                    "X-Trace-Id": trace_id,
                    "X-Job-Id": job_id,
                    "X-Proxy-Latency-Ms": str(int((time.time() - request.state.start_time) * 1000)) if hasattr(request.state, "start_time") else "0"
                }
            )

        except HTTPException:
            stats["failed_requests"] += 1
            raise

        except Exception as e:
            stats["failed_requests"] += 1
            log.error(f"代理失败: {e}", trace_id=trace_id, job_id=job_id)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            stats["current_concurrent"] -= 1


@app.middleware("http")
async def add_timing(request: Request, call_next):
    """添加请求时间追踪"""
    request.state.start_time = time.time()
    response = await call_next(request)
    return response


def main():
    import argparse

    parser = argparse.ArgumentParser(description="VLM Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8001, help="监听端口")
    parser.add_argument("--workers", type=int, default=1, help="Worker 数量")
    args = parser.parse_args()

    log.info(f"启动 VLM Proxy: {args.host}:{args.port}")
    uvicorn.run(
        "workers.vlm_proxy:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
