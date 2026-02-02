#!/usr/bin/env python3
"""
机动车违法检测分析器 (mv_violation)

检测目标:
- 闯红灯
- 违法变道
- 压线行驶
- 逆行
- 违法停车
- 不礼让行人

特点:
- 不使用 YOLO 检测，纯 Embedding + VLM
- 使用专门的违法模板进行语义匹配
"""
import os
import sys
import re
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.semantic_base import SemanticAnalyzerBase, AnalysisResult


# 机动车违法语义模板 (扩展版 - 60+ 模板)
MV_VIOLATION_TEMPLATES = [
    # === 闯红灯 (Red Light Running) ===
    "car running red light",
    "vehicle ignoring traffic signal",
    "car crossing on red",
    "running a red light at intersection",
    "vehicle passing through red traffic light",
    "car entering intersection on red signal",
    "red light violation",
    "traffic signal violation by car",
    "vehicle failing to stop at red light",
    "car ignoring stop signal",

    # === 违法变道 (Illegal Lane Change) ===
    "illegal lane change",
    "unsafe lane change",
    "car cutting across multiple lanes",
    "vehicle changing lanes without signal",
    "lane change across solid line",
    "cutting off other vehicles",
    "aggressive lane change",
    "vehicle crossing solid white line",
    "car weaving through traffic",
    "sudden lane change without indication",
    "dangerous overtaking maneuver",

    # === 压线行驶 (Lane Line Violation) ===
    "car crossing lane markings",
    "vehicle on lane line",
    "driving on road markings",
    "car straddling lanes",
    "vehicle driving on solid line",
    "car partially in two lanes",
    "lane marking violation",
    "vehicle touching lane divider",

    # === 逆行 (Wrong Way Driving) ===
    "wrong way driving",
    "car driving against traffic",
    "vehicle going opposite direction",
    "driving on wrong side of road",
    "vehicle in oncoming lane",
    "car heading into traffic",
    "contraflow driving",
    "vehicle traveling wrong direction",
    "driving against traffic flow",

    # === 违法停车 (Illegal Parking) ===
    "illegal parking",
    "car parked in no parking zone",
    "vehicle blocking traffic",
    "double parking",
    "parking on sidewalk",
    "vehicle in emergency lane",
    "car blocking crosswalk",
    "parking in bus lane",
    "stopping in intersection",
    "vehicle in no stopping zone",

    # === 不礼让行人 (Failing to Yield to Pedestrians) ===
    "car not yielding to pedestrian",
    "vehicle crossing crosswalk with pedestrians",
    "car ignoring pedestrian crossing",
    "not stopping for pedestrian",
    "vehicle entering zebra crossing with people",
    "car rushing through pedestrian crossing",
    "failing to yield at crosswalk",
    "vehicle endangering pedestrians",
    "car passing pedestrians on crosswalk",

    # === 超速 (Speeding - 通用) ===
    "vehicle speeding",
    "car driving too fast",
    "excessive speed",
    "high speed driving",

    # === 压黄线 (Yellow Line Violation) ===
    "crossing yellow line",
    "vehicle over yellow marking",
    "car on yellow center line"
]


class MVViolationAnalyzer(SemanticAnalyzerBase):
    """机动车违法检测分析器"""

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="mv_violation",
            consumer_group="mv_violation_workers",
            **kwargs
        )

    def get_templates(self) -> List[str]:
        return MV_VIOLATION_TEMPLATES

    def get_vlm_prompt(self) -> str:
        return """分析这些视频帧，判断是否存在机动车交通违法行为。

请检测以下违法类型:
1. 闯红灯 - 车辆在红灯亮起时通过路口
2. 违法变道 - 不打转向灯变道、跨越实线变道、连续变道
3. 压线行驶 - 车辆长时间骑压道路标线
4. 逆行 - 车辆在单行道或分隔带逆向行驶
5. 违法停车 - 在禁停区域停车、占用应急车道
6. 不礼让行人 - 在人行横道前不减速让行

请按以下 JSON 格式回答:
{
    "judgment": "YES/NO/UNCERTAIN",
    "confidence": 0.0-1.0,
    "violation_type": "闯红灯/违法变道/压线行驶/逆行/违法停车/不礼让行人/无",
    "reason": "简要说明判断依据"
}

注意事项:
- 仔细观察交通信号灯状态
- 注意道路标线（实线/虚线）
- 观察车辆与行人的相对位置
- 区分正常行驶和违法行为"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析 VLM 响应"""
        # 尝试解析 JSON
        import json

        # 清理响应文本
        response_text = response_text.strip()

        # 提取 JSON 部分
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return AnalysisResult(
                    judgment=data.get("judgment", "UNCERTAIN"),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", response_text),
                    violation_type=data.get("violation_type")
                )
            except json.JSONDecodeError:
                pass

        # 回退到简单解析
        judgment = "UNCERTAIN"
        confidence = 0.5
        violation_type = None

        upper_text = response_text.upper()[:100]
        if "YES" in upper_text:
            judgment = "YES"
            confidence = 0.8
            # 尝试提取违法类型
            for vt in ["闯红灯", "违法变道", "压线行驶", "逆行", "违法停车", "不礼让行人"]:
                if vt in response_text:
                    violation_type = vt
                    break
        elif "NO" in upper_text:
            judgment = "NO"
            confidence = 0.7

        return AnalysisResult(
            judgment=judgment,
            confidence=confidence,
            reason=response_text,
            violation_type=violation_type
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="机动车违法检测分析器")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    analyzer = MVViolationAnalyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
