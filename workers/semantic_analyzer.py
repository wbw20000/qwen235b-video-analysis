#!/usr/bin/env python3
"""
Semantic Analyzer - 事故检测分析器
继承 SemanticAnalyzerBase，获得 CUDA MOG2 运动检测 + FFmpeg NVDEC 解码能力

架构角色: 唯一的 accident orchestrator
消费 video_tasks 队列 → Embedding → 聚类 → VLM → result_tasks
"""
import os
import sys
import re
import json
from typing import List
from urllib.parse import quote

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.semantic_base import SemanticAnalyzerBase, AnalysisResult


# 事故模板（用于 SigLIP 相似度匹配）- 中文模板
ACCIDENT_TEMPLATES = [
    # 机动车之间事故
    "路口画面两辆机动车发生碰撞",
    "监控视频中汽车之间发生碰撞",
    "机动车追尾前车",
    "两车相撞，车辆受损",
    "十字路口车辆侧面碰撞",
    # 机动车与二轮车事故
    "路口画面机动车与二轮车发生碰撞",
    "监控视频中汽车与电动车发生碰撞",
    "机动车与自行车发生接触",
    "汽车与摩托车碰撞后骑车人摔倒",
    "路口机动车撞到电动车",
    # 机动车与行人事故
    "路口画面机动车与行人发生碰撞",
    "监控视频中汽车撞到行人",
    "机动车在人行横道撞到行人",
    "车辆与行人发生交通事故",
    # 多车事故
    "路口画面多车连环相撞",
    "监控视频中三辆或以上车辆连环相撞",
    "多车连撞事故",
    "车辆连环追尾事故",
    # 肇事逃逸
    "发生交通事故后车辆逃离现场",
    "肇事车辆逃逸",
    "碰撞后未停车离开",
    "事故后司机驾车逃逸",
]


class SemanticAnalyzer(SemanticAnalyzerBase):
    """事故检测分析器 - 继承基类获得 CUDA MOG2 + FFmpeg NVDEC"""

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="accident",
            consumer_group="semantic_analyzers",
            **kwargs
        )

    def get_templates(self) -> List[str]:
        return ACCIDENT_TEMPLATES

    def get_vlm_prompt(self) -> str:
        return """分析这些视频帧，判断是否发生了交通事故。

请按以下格式回答：
1. 判断: YES（确定发生事故）/ NO（未发生事故）/ UNCERTAIN（不确定）
2. 置信度: 0.0-1.0
3. 原因: 简要说明判断依据

注意事项：
- 观察车辆位置、姿态、运动轨迹
- 注意碰撞痕迹、车辆变形、人员倒地等迹象
- 区分正常行驶和事故场景"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析 VLM 响应 (YES/NO/UNCERTAIN)"""
        response_text = response_text.strip()

        # 尝试 JSON 解析
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return AnalysisResult(
                    judgment=data.get("judgment", "UNCERTAIN"),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", response_text)
                )
            except json.JSONDecodeError:
                pass

        # 回退到简单解析
        judgment = "UNCERTAIN"
        confidence = 0.5

        upper_text = response_text.upper()[:50]
        if "YES" in upper_text:
            judgment = "YES"
            confidence = 0.8
        elif "NO" in upper_text:
            judgment = "NO"
            confidence = 0.7

        return AnalysisResult(
            judgment=judgment,
            confidence=confidence,
            reason=response_text
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Analyzer - 事故检测")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    analyzer = SemanticAnalyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
