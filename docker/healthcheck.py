#!/usr/bin/env python3
"""通用健康检查脚本"""
import sys
import urllib.request
import urllib.error


def check_http(url: str, timeout: int = 5) -> bool:
    """检查 HTTP 端点"""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: healthcheck.py <url>")
        sys.exit(1)

    url = sys.argv[1]
    if check_http(url):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
