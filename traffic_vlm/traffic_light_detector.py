from __future__ import annotations

from typing import Dict, List


class TrafficLightDetector:
    """
    信号灯检测占位。当前返回空列表，保留接口以便未来接入 RSU/模型。
    """

    def __init__(self):
        pass

    def detect(self) -> List[Dict]:
        return []
