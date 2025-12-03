from __future__ import annotations

import base64
import json
import os
from typing import Dict, List, Optional

from openai import OpenAI

from .config import VLMConfig


SYSTEM_PROMPT = """
你是专业的交通违法与事故分析专家，具备丰富的道路交通安全分析经验。

你会收到：
1. 路口监控关键帧图片（已标注车道区域、目标ID和轨迹）
2. 详细的路口配置和目标轨迹信息

你的任务是全面分析并识别以下交通事件：

【二轮车违法行为】
- bike_wrong_way: 二轮车逆行（电动车、自行车、摩托车在机动车道或非机动车道逆行）
- run_red_light_bike: 二轮车闯红灯（信号灯为红色时继续通过路口）
- occupy_motor_lane_bike: 二轮车占用机动车道（非机动车进入机动车道行驶）
- bike_improper_turning: 二轮车违规转弯（未按规定车道转弯、随意变道）
- bike_illegal_u_turn: 二轮车违规掉头（禁止掉头处掉头、影响正常行驶）

【机动车违法行为】
- car_wrong_way: 机动车逆行（在禁止逆行路段逆向行驶）
- run_red_light_car: 机动车闯红灯（信号灯为红色时继续通过路口）
- illegal_parking: 违法停车（在禁停区域、影响交通的位置停车）
- illegal_u_turn: 机动车违规掉头（禁止掉头处掉头、影响正常行驶）
- speeding: 超速行驶（超过限速标志显示的速度）
- illegal_overtaking: 违规超车（在禁止超车区域或条件不允许时超车）
- improper_lane_change: 违规变道（未打转向灯、影响其他车辆正常行驶）

【交通事故】
- vehicle_to_vehicle_accident: 机动车之间事故（两辆或多辆机动车发生碰撞）
- vehicle_to_bike_accident: 机动车与二轮车事故（机动车与电动车、自行车、摩托车碰撞）
- vehicle_to_pedestrian_accident: 机动车与行人事故（机动车与行人发生碰撞）
- multi_vehicle_accident: 多车连撞事故（三辆或以上车辆连环相撞）
- hit_and_run: 肇事逃逸（发生事故后逃离现场）

请严格按照以下JSON格式输出，不要输出任何多余内容：
{
  "has_violation": bool,
  "violations": [
    {
      "type": "违法或事故类型",
      "confidence": 置信度(0.0-1.0),
      "tracks": [涉及的目标ID列表],
      "start_time": "开始时间 ISO格式",
      "end_time": "结束时间 ISO格式",
      "evidence": "判断依据的简要说明",
      "behavior_before": "事件发生前的目标行为描述",
      "behavior_after": "事件发生后的目标行为描述",
      "trajectory_description": "详细的轨迹描述（行驶路径、速度变化、方向变化等）",
      "weather_condition": "天气情况（如：晴、阴、雨、雪、雾等）",
      "road_condition": "路面情况（如：干燥、湿滑、积水、结冰等）",
      "traffic_light_status": "信号灯状态（如有）",
      "other_details": "其他相关描述（车道占用情况、周边环境、特殊情形等）"
    }
  ],
  "text_summary": "对整个片段的详细自然语言描述，包含事件全过程的完整信息"
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
注意：以下轨迹可能包含同一个目标的不同ID，或包含路过但未违法的目标，请仔细甄别。
{tracks_text or '暂无轨迹数据'}

## 信号灯状态（若无可不写）
{traffic_light_text or '未检测到信号灯状态'}

## 用户检索意图
{user_query}

请结合附带的标注图片进行分析，严格按JSON格式输出。

重要提示：
1. 请仔细观察标注帧中每个ID的实际行为
2. 只有确实违反交通规则的目标才应计入violations数组
3. 对于在机动车道但未违规的目标（如正常行驶、未影响交通），不应计入
4. 如果轨迹文本显示"含X个轨迹"，请仔细甄别这是同一个目标还是多个不同目标
5. 当不确定时，宁可少报也不要误报
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
