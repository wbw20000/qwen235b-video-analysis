#!/usr/bin/env python3
"""
电动自行车违法检测分析器 (ebike_violation)

检测目标:
- 闯红灯
- 逆行
- 载人超员
- 不戴头盔
- 占用机动车道
- 违法载货

特点:
- 专门针对电动自行车/二轮车的违法行为
- 不使用 YOLO 检测，纯 Embedding + VLM
"""
import os
import sys
import re
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.semantic_base import SemanticAnalyzerBase, AnalysisResult


# 电动自行车违法语义模板 (扩展版 - 70+ 模板)
EBIKE_VIOLATION_TEMPLATES = [
    # === 闯红灯 (Red Light Running) ===
    "electric scooter running red light",
    "e-bike ignoring traffic signal",
    "bicycle crossing on red",
    "two-wheeler running red light",
    "motorcycle passing red light",
    "scooter violating traffic signal",
    "cyclist running red light",
    "motorbike ignoring stop signal",

    # === 逆行 (Wrong Way Driving) ===
    "e-bike driving wrong way",
    "electric scooter against traffic",
    "bicycle on wrong side of road",
    "two-wheeler going opposite direction",
    "motorcycle driving against flow",
    "scooter in oncoming lane",
    "cyclist traveling wrong direction",
    "e-bike on contraflow",

    # === 载人超员 (Overloading Passengers) ===
    "e-bike with multiple passengers",
    "electric scooter carrying extra person",
    "overloaded bicycle with passengers",
    "two-wheeler with too many riders",
    "motorcycle with three people",
    "e-bike with two passengers",
    "scooter overloaded with people",
    "two people on electric bicycle",
    "multiple riders on scooter",
    "e-bike carrying adult passenger",

    # === 不戴头盔 (No Helmet) - 重点扩展 ===
    "e-bike rider without helmet",
    "electric scooter no helmet",
    "unhelmeted cyclist",
    "motorcycle rider not wearing helmet",
    "rider without safety helmet",
    "no helmet on e-bike",
    "scooter driver without helmet",
    "cyclist not wearing helmet",
    "bare head motorcycle rider",
    "e-bike passenger no helmet",
    "unhelmeted scooter rider",
    "person on e-bike without helmet",
    "helmet violation on motorcycle",
    "motorcycle without helmet",
    "electric bike no helmet",
    "two-wheeler rider no protective gear",
    "missing helmet on e-bike",
    "rider head unprotected",
    "no helmet electric scooter",
    "scooter passenger without helmet",

    # === 占用机动车道 (Illegal Lane Use) ===
    "e-bike in car lane",
    "electric scooter on motorway",
    "bicycle in vehicle lane",
    "two-wheeler blocking traffic lane",
    "motorcycle in fast lane",
    "e-bike on highway",
    "scooter in main road",
    "cyclist on motor vehicle lane",
    "e-bike in middle lane",
    "two-wheeler in car-only lane",

    # === 违法载货 (Illegal Cargo) ===
    "overloaded e-bike with cargo",
    "electric scooter carrying large items",
    "bicycle with oversized load",
    "two-wheeler with dangerous cargo",
    "e-bike with excessive cargo",
    "scooter carrying bulky items",
    "motorcycle with unstable load",
    "e-bike hauling furniture",
    "overloaded delivery scooter",

    # === 打伞骑行 (Umbrella While Riding) ===
    "e-bike rider holding umbrella",
    "cyclist with umbrella",
    "scooter rider using umbrella",
    "one-handed riding with umbrella",

    # === 其他违法 ===
    "e-bike running on sidewalk",
    "scooter on pedestrian path",
    "electric bike phone while riding"
]


class EbikeViolationAnalyzer(SemanticAnalyzerBase):
    """电动自行车违法检测分析器"""

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="ebike_violation",
            consumer_group="ebike_violation_workers",
            **kwargs
        )

    def get_templates(self) -> List[str]:
        return EBIKE_VIOLATION_TEMPLATES

    def get_vlm_prompt(self) -> str:
        return """分析这些视频帧，判断是否存在电动自行车/二轮车交通违法行为。

请检测以下违法类型:
1. 闯红灯 - 电动车/自行车在红灯时通过路口
2. 逆行 - 电动车在机动车道或非机动车道逆向行驶
3. 载人超员 - 电动自行车载人超过规定人数（成年人载一名儿童以上）
4. 不戴头盔 - 电动车驾驶人或乘客未佩戴安全头盔
5. 占用机动车道 - 电动车在机动车道行驶
6. 违法载货 - 载货超出车身范围或影响通行安全

请按以下 JSON 格式回答:
{
    "judgment": "YES/NO/UNCERTAIN",
    "confidence": 0.0-1.0,
    "violation_type": "闯红灯/逆行/载人超员/不戴头盔/占用机动车道/违法载货/无",
    "reason": "简要说明判断依据"
}

注意事项:
- 仔细识别电动车/自行车/摩托车
- 观察骑行人员是否佩戴头盔
- 注意载人情况和载货情况
- 观察行驶车道位置"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析 VLM 响应"""
        import json

        response_text = response_text.strip()
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

        judgment = "UNCERTAIN"
        confidence = 0.5
        violation_type = None

        upper_text = response_text.upper()[:100]
        if "YES" in upper_text:
            judgment = "YES"
            confidence = 0.8
            for vt in ["闯红灯", "逆行", "载人超员", "不戴头盔", "占用机动车道", "违法载货"]:
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

    parser = argparse.ArgumentParser(description="电动自行车违法检测分析器")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    analyzer = EbikeViolationAnalyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
