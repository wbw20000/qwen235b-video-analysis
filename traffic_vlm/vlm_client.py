from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from .config import VLMConfig


SYSTEM_PROMPT = """
你是交通违法分析专家。
你会收到：
1. 路口监控关键帧图片（已标注车道区域、目标ID和轨迹）
2. 简要的路口配置和目标轨迹信息
你的任务是判断是否存在以下违法行为：
- bike_wrong_way: 二轮车逆行
- run_red_light: 闯红灯（只有在明确信号灯为红色且车辆仍在通过时才判定）
- occupy_motor_lane: 二轮车占用机动车道
- accident: 交通事故（车辆/行人之间的碰撞或明显摔倒）

请严格按照以下JSON格式输出，不要输出任何多余内容：
{
  "has_violation": bool,
  "violations": [
    {
      "type": "violation_type",
      "confidence": 0.0-1.0,
      "tracks": [track_ids],
      "start_time": "ISO时间（若无法确定可为空字符串）",
      "end_time": "ISO时间（若无法确定可为空字符串）",
      "evidence": "简要说明判断依据"
    }
  ],
  "text_summary": "对该片段的自然语言描述"
}
"""


def image_to_base64_url(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


class VLMClient:
    def __init__(self, config: VLMConfig, api_key: Optional[str] = None):
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用云端 VLM")
        base_url = os.getenv("DASHSCOPE_BASE_URL")
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.config = config

    def build_user_prompt(
        self,
        intersection_info: Dict,
        tracks_text: str,
        traffic_light_text: str,
        user_query: str,
    ) -> str:
        return f"""
## 路口配置
- 路口类型: {intersection_info.get('intersection_type', '未知')}
- 车道方向说明: {intersection_info.get('direction_description', '未提供')}
- 非机动车道位置: {intersection_info.get('bike_lane_description', '未提供')}

## 结构化轨迹概要（仅供参考）
{tracks_text or '暂无轨迹数据'}

## 信号灯状态（若无可不写）
{traffic_light_text or '未检测到信号灯状态'}

## 用户检索意图
{user_query}

请结合附带的标注图片进行分析，严格按JSON格式输出。
"""

    def analyze(
        self,
        annotated_images: List[str],
        intersection_info: Dict,
        tracks_text: str,
        traffic_light_text: str,
        user_query: str,
    ) -> Dict:
        user_prompt = self.build_user_prompt(intersection_info, tracks_text, traffic_light_text, user_query)
        contents: List[Dict] = [{"type": "text", "text": user_prompt}]
        for img_path in annotated_images[: self.config.annotated_frames_per_clip]:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path)},
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        text = completion.choices[0].message.content
        try:
            # 尝试提取 JSON
            parsed = json.loads(text)
        except Exception:
            parsed = {"has_violation": False, "violations": [], "text_summary": text}
        return parsed
