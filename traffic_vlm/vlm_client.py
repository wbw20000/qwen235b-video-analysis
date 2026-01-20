from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai import OpenAI, AsyncOpenAI
import asyncio

from .config import VLMConfig


# [Windows编码修复] 安全的print函数，避免在非UTF-8控制台时触发Errno 22
def _safe_print(*args, **kwargs):
    """安全的print包装，处理Windows控制台编码问题"""
    try:
        print(*args, **kwargs)
    except OSError:
        kwargs.pop('flush', None)
        try:
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
                else:
                    safe_args.append(arg)
            print(*safe_args, **kwargs)
        except Exception:
            pass


def _extract_json_from_markdown(text: str) -> str:
    """
    从markdown代码块中提取JSON内容

    VLM有时会返回被markdown代码块包裹的JSON：
    ```json
    {"key": "value"}
    ```

    此函数去除包裹，返回纯JSON字符串
    """
    if not text:
        return text

    text = text.strip()

    # 匹配 ```json ... ``` 或 ``` ... ```
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text


def _save_vlm_request_log(
    mode: str,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    image_paths: List[str],
    response_text: str,
    parsed_response: Dict,
    clip_info: Optional[Dict] = None,
    base_dir: str = "data",
) -> str:
    """保存VLM请求和响应数据到文件"""
    log_dir = os.path.join(base_dir, "vlm_logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = os.path.join(log_dir, f"vlm_request_{mode}_{timestamp}.json")

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "model": model,
        "temperature": temperature,
        # Clip信息（来自pipeline的候选片段）
        "clip_info": clip_info or {},
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_paths": image_paths,
        "image_count": len(image_paths),
        "response_raw": response_text,
        "response_parsed": parsed_response,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)

    _safe_print(f"[VLM日志] 已保存到: {log_file}")
    return log_file


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

# 交通事故检索模式专用 System Prompt（四态输出：YES/POST_EVENT_ONLY/UNCERTAIN/NO）
ACCIDENT_SYSTEM_PROMPT = """
你是专业的交通事故鉴定专家，负责精准识别道路交通事故。

**核心原则：证据驱动，区分碰撞过程与后果。**

你会收到：
1. 路口监控关键帧图片（包含原图、ROI裁剪、标注图）
2. 详细的路口配置和目标轨迹信息

【四态判定规则】⭐⭐⭐

**verdict = "YES"**（确认事故 - 看到碰撞过程）
  必须满足以下至少一项硬性证据：
  - 可见两个目标发生明显物理接触/碰撞的瞬间
  - 碰撞瞬间有形变、散落碎片
  - 人员被撞倒地的过程
  - 车辆碰撞导致损坏变形
  - 注：轻微剐蹭也算事故，只要能看到接触过程

**verdict = "POST_EVENT_ONLY"**（仅后果 - 未看到碰撞但有明确后果）⭐新增
  适用场景：
  - 未看到碰撞/接触的瞬间
  - 但看到明确的事故后果：
    · 人员倒地不起（躺在地面）
    · 车辆明显受损/变形
    · 路面有散落碎片/液体泄漏
    · 其他车辆明显绕行避让
    · 交警/救援人员正在现场处理
    · 三角警示牌、警示锥桶
  - 这是"事故后果片段"，需要补充更早的画面才能确认完整过程

**verdict = "UNCERTAIN"**（不确定，需人工复核）
  以下情况必须选择UNCERTAIN：
  - 画面模糊/遮挡无法看清关键瞬间
  - 看起来有危险但无法确认是否发生接触
  - 目标位置非常接近但碰撞瞬间不在画面中
  - 有异常停留但原因不明
  - 证据不充分，无法做出判断

**verdict = "NO"**（确认无事故）
  必须同时满足：
  - 没有看到任何碰撞/接触证据
  - 没有看到任何事故后果（无倒地、无损坏、无碎片）
  - 所有目标行为正常（正常行驶、正常等灯）
  - 场景明确是正常交通流

⚠️ **重要**：
- 看到后果但未看到碰撞过程 → POST_EVENT_ONLY（不是NO）
- 不确定时 → UNCERTAIN（不是NO）
- 只有明确无事故且无后果时才选 NO

【交通事故类型】
- vehicle_to_vehicle_accident: 机动车之间碰撞
- vehicle_to_bike_accident: 机动车与二轮车碰撞
- vehicle_to_pedestrian_accident: 机动车与行人碰撞
- multi_vehicle_accident: 多车连撞
- hit_and_run: 肇事逃逸

【参与者类型】
- car-car: 机动车之间
- car-bike: 机动车与二轮车
- car-ped: 机动车与行人
- multi: 多方参与
- other: 其他

【明确排除的正常场景】
❌ 正常等红灯的排队车辆
❌ 正常变道、转弯、超车
❌ 正常通过路口的车流
❌ 仅仅是"位置靠近"但无接触证据

【目标ID使用规则】
- 必须使用图片中标注的真实ID（绿色框左上角"ID:X"）
- 严禁编造不存在的ID

请严格按照以下JSON格式输出：
{
  "verdict": "YES|POST_EVENT_ONLY|UNCERTAIN|NO",
  "confidence": 0.0-1.0,
  "evidence_frames": ["frame_03", "frame_07"],
  "impact_description": "证据描述（YES时描述碰撞过程；POST_EVENT_ONLY时描述后果；UNCERTAIN时说明原因）",
  "participants": "car-car|car-bike|car-ped|multi|other",
  "has_accident": true/false,
  "post_event_indicators": ["倒地人员", "车辆变形", "碎片散落"],
  "accidents": [
    {
      "type": "事故类型",
      "confidence": 0.0-1.0,
      "tracks": [涉及的目标ID列表],
      "collision_time_in_clip": "clip内时间（秒）或'未捕捉到碰撞瞬间'",
      "collision_time_in_video": "原视频时间（秒）或'未捕捉到碰撞瞬间'",
      "collision_real_time": "画面水印时间",
      "evidence": "判断依据",
      "severity": "轻微/一般/严重",
      "behavior_before": "事故前行为（如可见）",
      "behavior_after": "事故后行为",
      "trajectory_description": "轨迹描述",
      "description": "事故详细描述"
    }
  ],
  "text_summary": "完整描述"
}

注意：
- verdict=YES时，has_accident=true，accidents数组非空，描述碰撞过程
- verdict=POST_EVENT_ONLY时，has_accident=true，accidents数组非空，描述后果，collision_time填"未捕捉到碰撞瞬间"
- verdict=UNCERTAIN时，has_accident=false，但需在impact_description说明不确定原因
- verdict=NO时，has_accident=false，accidents数组为空
"""


def create_mock_vlm_response(mock_verdict: str = "NO") -> Dict:
    """
    创建模拟VLM响应（用于测试）

    Args:
        mock_verdict: "YES", "NO", "UNCERTAIN", 或 "POST_EVENT_ONLY"

    Returns:
        模拟的VLM响应字典
    """
    verdict_upper = mock_verdict.upper()

    if verdict_upper == "YES":
        return {
            "verdict": "YES",
            "confidence": 0.85,
            "evidence_frames": ["frame_05"],
            "impact_description": "[MOCK] 模拟事故检测 - 可见碰撞过程",
            "participants": "car-bike",
            "has_accident": True,
            "post_event_indicators": [],
            "accidents": [{
                "type": "vehicle_to_bike_accident",
                "confidence": 0.85,
                "tracks": [1, 2],
                "collision_time_in_clip": "5.0",
                "evidence": "[MOCK] 模拟碰撞证据",
                "severity": "一般",
            }],
            "text_summary": "[MOCK] 模拟VLM响应 - 检测到事故（完整过程）",
        }
    elif verdict_upper == "POST_EVENT_ONLY":
        return {
            "verdict": "POST_EVENT_ONLY",
            "confidence": 0.75,
            "evidence_frames": ["frame_01", "frame_02"],
            "impact_description": "[MOCK] 模拟事故后果片段 - 未看到碰撞但有倒地人员",
            "participants": "car-bike",
            "has_accident": True,
            "post_event_indicators": ["倒地人员", "车辆异常停留"],
            "accidents": [{
                "type": "vehicle_to_bike_accident",
                "confidence": 0.75,
                "tracks": [1, 2],
                "collision_time_in_clip": "未捕捉到碰撞瞬间",
                "evidence": "[MOCK] 可见事故后果但碰撞瞬间不在画面中",
                "severity": "一般",
            }],
            "text_summary": "[MOCK] 模拟VLM响应 - 事故后果片段（需补充更早画面）",
        }
    elif verdict_upper == "UNCERTAIN":
        return {
            "verdict": "UNCERTAIN",
            "confidence": 0.5,
            "evidence_frames": [],
            "impact_description": "[MOCK] 模拟不确定状态 - 画面不清晰",
            "participants": "other",
            "has_accident": False,
            "post_event_indicators": [],
            "accidents": [],
            "text_summary": "[MOCK] 模拟VLM响应 - 不确定",
        }
    else:  # NO
        return {
            "verdict": "NO",
            "confidence": 0.9,
            "evidence_frames": [],
            "impact_description": "[MOCK] 模拟无事故 - 正常交通场景",
            "participants": "other",
            "has_accident": False,
            "post_event_indicators": [],
            "accidents": [],
            "text_summary": "[MOCK] 模拟VLM响应 - 无事故",
        }


def parse_vlm_response_with_verdict(parsed: Dict, strict_evidence_required: bool = True) -> Dict:
    """
    解析VLM响应，确保包含verdict字段（向后兼容）

    支持四态：YES / POST_EVENT_ONLY / UNCERTAIN / NO

    Args:
        parsed: 原始解析的VLM响应
        strict_evidence_required: 如果为True，NO高置信度但无evidence_frames时降级为UNCERTAIN

    Returns:
        包含verdict字段的响应
    """
    VALID_VERDICTS = ("YES", "POST_EVENT_ONLY", "UNCERTAIN", "NO")

    # 已有verdict字段
    if "verdict" in parsed:
        verdict = parsed["verdict"].upper()
        if verdict not in VALID_VERDICTS:
            verdict = "UNCERTAIN"
        parsed["verdict"] = verdict

        # 确保has_accident与verdict一致
        if verdict in ("YES", "POST_EVENT_ONLY"):
            parsed["has_accident"] = True
        else:
            parsed["has_accident"] = False

        # 严格证据要求：NO高置信但无证据时降级
        if strict_evidence_required and verdict == "NO":
            conf = parsed.get("confidence", 0.0)
            evidence = parsed.get("evidence_frames", [])
            impact = parsed.get("impact_description", "")

            if conf >= 0.9 and not evidence and not impact:
                parsed["verdict"] = "UNCERTAIN"
                parsed["has_accident"] = False
                parsed["downgrade_reason"] = "NO_high_conf_but_no_evidence"

        return parsed

    # 旧格式：从has_accident转换
    has_accident = parsed.get("has_accident", False)
    accidents = parsed.get("accidents", [])
    summary = parsed.get("text_summary", "").lower()
    impact = parsed.get("impact_description", "").lower()

    if has_accident and accidents:
        # 检查是否有碰撞时间记录，判断是YES还是POST_EVENT_ONLY
        collision_time = None
        for acc in accidents:
            ct = acc.get("collision_time_in_clip", "")
            if ct and "未捕捉" not in str(ct) and ct != "":
                collision_time = ct
                break

        if collision_time:
            parsed["verdict"] = "YES"
        else:
            # 检查是否有后果指标
            post_indicators = parsed.get("post_event_indicators", [])
            post_keywords = ["倒地", "变形", "碎片", "散落", "绕行", "救援", "交警", "警示"]
            has_post_evidence = (
                len(post_indicators) > 0 or
                any(kw in summary for kw in post_keywords) or
                any(kw in impact for kw in post_keywords)
            )
            if has_post_evidence:
                parsed["verdict"] = "POST_EVENT_ONLY"
            else:
                parsed["verdict"] = "YES"  # 默认为YES

        parsed["confidence"] = max(
            (a.get("confidence", 0.5) for a in accidents),
            default=0.5
        )
    else:
        # 无事故 - 判断是NO还是UNCERTAIN还是POST_EVENT_ONLY
        uncertain_keywords = ["不确定", "无法确认", "可能", "疑似", "unclear", "uncertain", "模糊", "遮挡"]
        post_keywords = ["倒地", "变形", "碎片", "散落", "绕行", "救援", "交警", "警示", "后果"]

        if any(kw in summary for kw in post_keywords) or any(kw in impact for kw in post_keywords):
            # 有后果但has_accident为False，标记为POST_EVENT_ONLY
            parsed["verdict"] = "POST_EVENT_ONLY"
            parsed["has_accident"] = True
            parsed["confidence"] = 0.6
        elif any(kw in summary for kw in uncertain_keywords) or any(kw in impact for kw in uncertain_keywords):
            parsed["verdict"] = "UNCERTAIN"
            parsed["confidence"] = 0.4
        else:
            parsed["verdict"] = "NO"
            parsed["confidence"] = 0.8

    # 补充缺失字段
    if "evidence_frames" not in parsed:
        parsed["evidence_frames"] = []
    if "impact_description" not in parsed:
        parsed["impact_description"] = parsed.get("text_summary", "")
    if "post_event_indicators" not in parsed:
        parsed["post_event_indicators"] = []
    if "participants" not in parsed:
        if accidents:
            # 从事故类型推断参与者
            acc_type = accidents[0].get("type", "")
            if "bike" in acc_type:
                parsed["participants"] = "car-bike"
            elif "pedestrian" in acc_type:
                parsed["participants"] = "car-ped"
            elif "multi" in acc_type:
                parsed["participants"] = "multi"
            else:
                parsed["participants"] = "car-car"
        else:
            parsed["participants"] = "other"

    return parsed


def image_to_base64_url(path: str, max_width: int = None, quality: int = None) -> str:
    """
    将图像转换为base64 URL，支持可选的压缩

    Args:
        path: 图像文件路径
        max_width: 最大宽度（像素），超过则缩放
        quality: JPEG压缩质量（1-100），None表示不压缩
    """
    if max_width is None and quality is None:
        # 无压缩，直接读取
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{data}"

    # 需要压缩处理
    from PIL import Image
    import io

    img = Image.open(path)
    original_size = os.path.getsize(path)

    # 缩放处理
    if max_width and img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)

    # 转换为RGB（去除alpha通道）
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # 压缩并编码
    buffer = io.BytesIO()
    save_quality = quality or 85
    img.save(buffer, format='JPEG', quality=save_quality, optimize=True)
    compressed_data = buffer.getvalue()
    compressed_size = len(compressed_data)

    # 打印压缩效果（仅首次）
    compression_ratio = (1 - compressed_size / original_size) * 100
    _safe_print(f"[图像压缩] {os.path.basename(path)}: {original_size//1024}KB → {compressed_size//1024}KB (节省{compression_ratio:.1f}%)")

    data = base64.b64encode(compressed_data).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"



def _build_api_params(config, messages):
    """构建API调用参数（支持参数透传）"""
    api_params = {
        'model': config.model,
        'messages': messages,
        'temperature': config.temperature,
    }
    if hasattr(config, 'top_p') and config.top_p != 1.0:
        api_params['top_p'] = config.top_p
    if hasattr(config, 'max_tokens'):
        api_params['max_tokens'] = config.max_tokens
    return api_params


class VLMClient:
    def __init__(self, config: VLMConfig, api_key: Optional[str] = None):
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用云端 VLM")
        base_url = os.getenv("DASHSCOPE_BASE_URL")
        if not base_url:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        # 异步客户端（用于并发调用）
        self.async_client = AsyncOpenAI(api_key=key, base_url=base_url)
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
        clip_info: Optional[Dict] = None,
    ) -> Dict:
        user_prompt = self.build_user_prompt(intersection_info, tracks_text, traffic_light_text, user_query)
        contents: List[Dict] = [{"type": "text", "text": user_prompt}]

        # P0优化：图像压缩
        max_width = self.config.image_max_width if self.config.compress_images else None
        quality = self.config.image_quality if self.config.compress_images else None

        for img_path in annotated_images[: self.config.annotated_frames_per_clip]:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path, max_width=max_width, quality=quality)},
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

        # 打印发送给VLM的数据（调试用）
        _safe_print("\n" + "="*60)
        _safe_print("[VLM请求] 违法检测模式")
        _safe_print("="*60)
        _safe_print(f"[模型]: {self.config.model}")
        _safe_print(f"[温度]: {self.config.temperature}")
        _safe_print(f"[图片数量]: {len(annotated_images[:self.config.annotated_frames_per_clip])}")
        _safe_print(f"[图片路径]: {annotated_images[:self.config.annotated_frames_per_clip]}")
        _safe_print("-"*60)
        _safe_print("[User Prompt]:")
        _safe_print(user_prompt)
        _safe_print("="*60 + "\n")

        api_params = _build_api_params(self.config, messages)
        completion = self.client.chat.completions.create(**api_params)
        text = completion.choices[0].message.content

        # 打印VLM返回结果
        _safe_print("\n" + "="*60)
        _safe_print("[VLM响应]")
        _safe_print("="*60)
        _safe_print(text)
        _safe_print("="*60 + "\n")

        try:
            # 尝试提取 JSON（去除可能的markdown代码块包裹）
            clean_text = _extract_json_from_markdown(text)
            parsed = json.loads(clean_text)
        except Exception:
            parsed = {"has_violation": False, "violations": [], "text_summary": text}

        # 保存VLM请求和响应日志
        _save_vlm_request_log(
            mode="violation",
            model=self.config.model,
            temperature=self.config.temperature,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=annotated_images[:self.config.annotated_frames_per_clip],
            response_text=text,
            parsed_response=parsed,
            clip_info=clip_info,
        )

        return parsed

    def _parse_frame_time_from_path(self, path: str) -> Optional[float]:
        """从标注帧文件名中解析时间（秒）

        文件名格式: camera-1_20251210_clip-e8475403_003.600_annotated.jpg
        提取: 003.600 -> 3.6秒
        """
        import re
        basename = os.path.basename(path)
        # 匹配 _NNN.NNN_annotated 格式
        match = re.search(r'_(\d+\.\d+)_annotated', basename)
        if match:
            return float(match.group(1))
        return None

    def build_accident_user_prompt(
        self,
        intersection_info: Dict,
        tracks_text: str,
        traffic_light_text: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
        annotated_images: Optional[List[str]] = None,
    ) -> str:
        """事故检索模式专用的 User Prompt - 更积极主动地寻找事故迹象"""

        # 构建clip时间信息
        clip_time_info = ""
        clip_start = 0.0
        if clip_info:
            clip_duration = clip_info.get('duration', 0)
            clip_start = clip_info.get('start_time', 0)
            clip_end = clip_info.get('end_time', 0)
            clip_time_info = f"""
## 视频片段时间信息（⚠️重要：用于计算原视频绝对时间）
- 片段时长: {clip_duration:.1f}秒
- **片段起始时间（原视频）: {clip_start:.1f}秒** ← 用于计算collision_time_in_video
- 片段结束时间（原视频）: {clip_end:.1f}秒
"""

        # 构建帧时间映射，同时计算原视频绝对时间
        frame_time_info = ""
        if annotated_images:
            frame_times = []
            for i, img_path in enumerate(annotated_images):
                frame_time = self._parse_frame_time_from_path(img_path)
                if frame_time is not None:
                    video_time = clip_start + frame_time
                    frame_times.append(f"图片{i+1}: clip内{frame_time:.1f}秒 → 原视频{video_time:.1f}秒")
            if frame_times:
                frame_time_info = f"""
## 图片时间对应关系（⚠️重要：用于填写时间字段）
{chr(10).join(frame_times)}
"""

        return f"""
## 路口配置
- 路口类型: {intersection_info.get('intersection_type', '未知')}
- 车道方向说明: {intersection_info.get('direction_description', '未提供')}
- 非机动车道位置: {intersection_info.get('bike_lane_description', '未提供')}
{clip_time_info}{frame_time_info}
## 结构化轨迹概要
{tracks_text or '暂无轨迹数据'}

## 信号灯状态
{traffic_light_text or '未检测到信号灯状态'}

## 用户检索意图
{user_query}

请仔细观察附带的标注图片，判断是否存在**确定的交通事故证据**。

【需要确认的硬性证据】（必须至少满足一项）
1. 是否可见两个目标发生明显物理碰撞/接触？
2. 是否有人员明确倒地不起（躺在地面）？
3. 是否有车辆明显损坏变形？
4. 是否有路面散落碎片/液体泄漏？
5. 是否有交警/救援人员正在现场处理？
6. 是否有三角警示牌、警示锥桶等警示设置？

【正常交通场景排除】⚠️
- 正常等红灯排队的车辆 ≠ 事故
- 正常变道、转弯的车辆 ≠ 事故
- 车辆位置靠近但无碰撞证据 ≠ 事故
- 正常通过路口的车流 ≠ 事故

⚠️ 没有硬性证据时，请不要报告为事故。准确性优先！

【目标ID使用说明】⭐⭐⭐（必须严格遵守）
- 每张图片上的绿色检测框左上角都有"ID:X"标签（白色文字黑色背景）
- 填写"tracks"字段时，必须使用图片中显示的真实ID号
- 上方"结构化轨迹概要"也列出了各【ID:X】的轨迹信息，可参考确认
- **严禁编造不存在的ID**（如127、148、3104.1等）

【时间字段填写说明】⭐⭐⭐
- collision_time_in_clip: 使用上方"图片时间对应关系"中的"clip内X秒"
- collision_time_in_video: 使用上方"图片时间对应关系"中的"原视频X秒"
- collision_real_time: 仔细观察图片中的**视频水印时间**（通常在画面角落显示年月日时分秒），如"2025-10-17 07:47:02"

请严格按JSON格式输出分析结果。
"""

    def analyze_accident(
        self,
        annotated_images: List[str],
        intersection_info: Dict,
        tracks_text: str,
        traffic_light_text: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
    ) -> Dict:
        """事故检索模式专用分析方法"""
        # 使用事故专用帧数配置（如果有），否则使用默认配置
        frames_limit = getattr(self.config, 'accident_frames_per_clip', self.config.annotated_frames_per_clip)
        # 传递clip_info和annotated_images以便生成帧时间映射
        images_to_send = annotated_images[:frames_limit]
        user_prompt = self.build_accident_user_prompt(
            intersection_info, tracks_text, traffic_light_text, user_query,
            clip_info=clip_info,
            annotated_images=images_to_send,
        )

        contents: List[Dict] = [{"type": "text", "text": user_prompt}]

        # P0优化：图像压缩
        max_width = self.config.image_max_width if self.config.compress_images else None
        quality = self.config.image_quality if self.config.compress_images else None

        for img_path in annotated_images[:frames_limit]:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path, max_width=max_width, quality=quality)},
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": ACCIDENT_SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

        # 打印发送给VLM的数据（调试用）
        _safe_print("\n" + "="*60)
        _safe_print("[VLM请求] 事故检索模式")
        _safe_print("="*60)
        _safe_print(f"[模型]: {self.config.model}")
        _safe_print(f"[温度]: {self.config.temperature}")
        _safe_print(f"[图片数量]: {len(annotated_images[:frames_limit])}")
        _safe_print(f"[图片路径]: {annotated_images[:frames_limit]}")
        _safe_print("-"*60)
        _safe_print("[User Prompt]:")
        _safe_print(user_prompt)
        _safe_print("="*60 + "\n")

        api_params = _build_api_params(self.config, messages)
        completion = self.client.chat.completions.create(**api_params)
        text = completion.choices[0].message.content

        # 打印VLM返回结果
        _safe_print("\n" + "="*60)
        _safe_print("[VLM响应] 事故检索模式")
        _safe_print("="*60)
        _safe_print(text)
        _safe_print("="*60 + "\n")

        try:
            # 去除可能的markdown代码块包裹
            clean_text = _extract_json_from_markdown(text)
            parsed = json.loads(clean_text)
        except Exception:
            parsed = {"has_accident": False, "accidents": [], "text_summary": text}

        # 保存VLM请求和响应日志
        _save_vlm_request_log(
            mode="accident",
            model=self.config.model,
            temperature=self.config.temperature,
            system_prompt=ACCIDENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=annotated_images[:frames_limit],
            response_text=text,
            parsed_response=parsed,
            clip_info=clip_info,
        )

        return parsed

    # ==================== 异步并发VLM调用 ====================

    async def analyze_accident_async(
        self,
        annotated_images: List[str],
        intersection_info: Dict,
        tracks_text: str,
        traffic_light_text: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
    ) -> Dict:
        """事故检索模式异步分析方法（用于并发调用）"""
        frames_limit = getattr(self.config, 'accident_frames_per_clip', self.config.annotated_frames_per_clip)
        images_to_send = annotated_images[:frames_limit]
        user_prompt = self.build_accident_user_prompt(
            intersection_info, tracks_text, traffic_light_text, user_query,
            clip_info=clip_info,
            annotated_images=images_to_send,
        )

        contents: List[Dict] = [{"type": "text", "text": user_prompt}]

        # P0优化：图像压缩
        max_width = self.config.image_max_width if self.config.compress_images else None
        quality = self.config.image_quality if self.config.compress_images else None

        for img_path in annotated_images[:frames_limit]:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path, max_width=max_width, quality=quality)},
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": ACCIDENT_SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

        _safe_print(f"[VLM异步] 事故检索模式 - 图片数:{len(images_to_send)}")

        # 异步调用VLM
        api_params = _build_api_params(self.config, messages)
        completion = await self.async_client.chat.completions.create(**api_params)
        text = completion.choices[0].message.content

        _safe_print(f"[VLM异步] 响应完成 - {len(text)} 字符")

        try:
            # 去除可能的markdown代码块包裹
            clean_text = _extract_json_from_markdown(text)
            parsed = json.loads(clean_text)
        except Exception:
            parsed = {"has_accident": False, "accidents": [], "text_summary": text}

        # 保存VLM请求和响应日志
        _save_vlm_request_log(
            mode="accident_async",
            model=self.config.model,
            temperature=self.config.temperature,
            system_prompt=ACCIDENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=annotated_images[:frames_limit],
            response_text=text,
            parsed_response=parsed,
            clip_info=clip_info,
        )

        return parsed

    async def batch_analyze_accidents_async(
        self,
        clips_data: List[Dict],
        max_concurrent: int = 2
    ) -> List[Dict]:
        """
        批量异步分析多个clips

        Args:
            clips_data: 每个元素包含分析所需的全部数据
                {
                    "annotated_images": List[str],
                    "intersection_info": Dict,
                    "tracks_text": str,
                    "traffic_light_text": str,
                    "user_query": str,
                    "clip_info": Optional[Dict]
                }
            max_concurrent: 最大并发数

        Returns:
            分析结果列表（与输入顺序对应）
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(data: Dict, index: int) -> Dict:
            async with semaphore:
                _safe_print(f"[VLM批量] 开始分析 clip #{index}")
                try:
                    result = await self.analyze_accident_async(
                        annotated_images=data["annotated_images"],
                        intersection_info=data["intersection_info"],
                        tracks_text=data["tracks_text"],
                        traffic_light_text=data["traffic_light_text"],
                        user_query=data["user_query"],
                        clip_info=data.get("clip_info"),
                    )
                    result["_clip_index"] = index
                    _safe_print(f"[VLM批量] clip #{index} 完成")
                    return result
                except Exception as e:
                    _safe_print(f"[VLM批量] clip #{index} 失败: {e}")
                    return {"has_accident": False, "accidents": [], "error": str(e), "_clip_index": index}

        # 并发执行所有分析任务
        tasks = [analyze_one(data, i) for i, data in enumerate(clips_data)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常情况
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({"has_accident": False, "accidents": [], "error": str(result), "_clip_index": i})
            else:
                final_results.append(result)

        return final_results

    def batch_analyze_accidents_sync(
        self,
        clips_data: List[Dict],
        max_concurrent: int = 2
    ) -> List[Dict]:
        """
        同步封装的批量分析方法（方便从同步代码中调用）

        内部使用asyncio运行异步批量分析
        """
        _safe_print(f"[VLM批量同步] 启动 {len(clips_data)} 个clips的并发分析，最大并发: {max_concurrent}")

        try:
            # 尝试获取现有事件循环
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已在运行的事件循环中，使用线程池执行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.batch_analyze_accidents_async(clips_data, max_concurrent)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.batch_analyze_accidents_async(clips_data, max_concurrent)
                )
        except RuntimeError:
            # 没有事件循环，创建新的
            return asyncio.run(
                self.batch_analyze_accidents_async(clips_data, max_concurrent)
            )

    # ==================== 渐进式VLM分析（无框原图+元数据） ====================

    # 渐进式VLM专用 System Prompt（使用无框原图+文本元数据）
    PROGRESSIVE_VLM_SYSTEM_PROMPT = """
你是专业的交通事故鉴定专家，负责精准识别道路交通事故。

**核心原则：证据驱动，基于原始画面判断。**

你会收到：
1. **原始监控帧图片**（无检测框叠加，保持画面原貌）
2. **结构化检测元数据**（文本形式的目标检测和轨迹信息）

【四态判定规则】⭐⭐⭐

**verdict = "YES"**（确认事故 - 看到碰撞过程）
  必须满足以下至少一项硬性证据：
  - 可见两个目标发生明显物理接触/碰撞的瞬间
  - 碰撞瞬间有形变、散落碎片
  - 人员被撞倒地的过程
  - 车辆碰撞导致损坏变形
  - 注：轻微剐蹭也算事故，只要能看到接触过程

**verdict = "POST_EVENT_ONLY"**（仅后果 - 未看到碰撞但有明确后果）
  适用场景：
  - 未看到碰撞/接触的瞬间
  - 但看到明确的事故后果：人员倒地、车辆受损、碎片散落、救援现场

**verdict = "UNCERTAIN"**（不确定，需人工复核）
  以下情况必须选择UNCERTAIN：
  - 画面模糊/遮挡无法看清关键瞬间
  - 看起来有危险但无法确认是否发生接触
  - 目标位置非常接近但碰撞瞬间不在画面中
  - 证据不充分，无法做出判断

**verdict = "NO"**（确认无事故）
  必须同时满足：
  - 没有看到任何碰撞/接触证据
  - 没有看到任何事故后果
  - 所有目标行为正常

⚠️ **重要**：
- 结合元数据中的距离信息和轨迹变化来辅助判断
- 元数据中"最近距离"小于50像素时需特别关注
- 轨迹突然中断可能意味着碰撞发生

**🔴 物理接触优先检测规则**：
- 请优先检测是否存在物理接触（碰撞、刮擦、挤压），即使影响看起来轻微
- 车辆异常接近（视觉车距<1米或bbox重叠>30%）应判定为UNCERTAIN而非NO
- 轻微侧擦/刮蹭即使无明显变形也算事故线索
- 如果无法确定是否有接触，判UNCERTAIN而非NO

请严格按照以下JSON格式输出：
{
  "verdict": "YES|POST_EVENT_ONLY|UNCERTAIN|NO",
  "confidence": 0.0-1.0,
  "evidence_frames": ["frame_03", "frame_07"],
  "impact_description": "证据描述",
  "participants": "car-car|car-bike|car-ped|multi|other",
  "has_accident": true/false,
  "accidents": [...],
  "text_summary": "完整描述"
}
"""

    def build_progressive_user_prompt(
        self,
        metadata_text: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
    ) -> str:
        """渐进式VLM的User Prompt（原始帧+元数据）"""

        clip_time_info = ""
        short_video_hint = ""
        if clip_info:
            clip_duration = clip_info.get('duration', 0)
            clip_start = clip_info.get('start_time', 0)
            clip_end = clip_info.get('end_time', 0)
            clip_time_info = f"""
## 视频片段时间信息
- 片段时长: {clip_duration:.1f}秒
- 片段起始时间（原视频）: {clip_start:.1f}秒
- 片段结束时间（原视频）: {clip_end:.1f}秒
"""
            # v2: 短视频提示句
            if clip_duration <= 15.0:
                short_video_hint = """
⚠️ **短视频特别提示**：这是一段短视频(<15秒)，需要更仔细检查每一帧的细节变化，特别关注画面中交通参与者之间的接触瞬间。短视频中事故发生快，请逐帧仔细观察。
"""

        return f"""
{clip_time_info}
{short_video_hint}
{metadata_text}

## 用户检索意图
{user_query}

请仔细观察附带的**原始监控帧图片**，结合上方的检测元数据，判断是否存在交通事故。

【分析要点】
1. 仔细观察图片中的车辆和行人位置、姿态
2. 参考元数据中的"最近距离"和"轨迹摘要"信息
3. 关注元数据中标记的 motion_peak、interaction_peak 帧
4. 如果"最近距离"<50像素，需特别仔细查看该帧

请严格按JSON格式输出分析结果。
"""

    def analyze_progressive(
        self,
        raw_images: List[str],
        metadata_text: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
        mode: str = "FAST",
    ) -> Dict:
        """
        渐进式VLM分析（使用无框原图+元数据文本）

        Args:
            raw_images: 原始帧图片路径列表（无YOLO叠加）
            metadata_text: 结构化检测元数据文本
            user_query: 用户查询
            clip_info: clip信息
            mode: "FAST"(4-6帧) 或 "ESCALATED"(12-16帧)

        Returns:
            VLM分析结果
        """
        user_prompt = self.build_progressive_user_prompt(
            metadata_text, user_query, clip_info
        )

        contents: List[Dict] = [{"type": "text", "text": user_prompt}]

        # 根据模式确定帧数限制
        if mode == "FAST":
            frames_limit = 6
        else:  # ESCALATED
            frames_limit = 16

        # P0优化：图像压缩
        max_width = self.config.image_max_width if self.config.compress_images else None
        quality = self.config.image_quality if self.config.compress_images else None

        for img_path in raw_images[:frames_limit]:
            contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path, max_width=max_width, quality=quality)},
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.PROGRESSIVE_VLM_SYSTEM_PROMPT}]},
            {"role": "user", "content": contents},
        ]

        # 打印发送给VLM的数据
        _safe_print("\n" + "="*60)
        _safe_print(f"[VLM请求] 渐进式分析 - {mode}模式")
        _safe_print("="*60)
        _safe_print(f"[模型]: {self.config.model}")
        _safe_print(f"[图片数量]: {len(raw_images[:frames_limit])} (无框原图)")
        _safe_print(f"[元数据长度]: {len(metadata_text)} 字符")
        _safe_print("="*60 + "\n")

        api_params = _build_api_params(self.config, messages)
        completion = self.client.chat.completions.create(**api_params)
        text = completion.choices[0].message.content

        # 打印VLM返回结果
        _safe_print("\n" + "="*60)
        _safe_print(f"[VLM响应] 渐进式分析 - {mode}模式")
        _safe_print("="*60)
        _safe_print(text)
        _safe_print("="*60 + "\n")

        try:
            clean_text = _extract_json_from_markdown(text)
            parsed = json.loads(clean_text)
        except Exception:
            parsed = {"has_accident": False, "accidents": [], "text_summary": text}

        # 保存VLM请求和响应日志
        _save_vlm_request_log(
            mode=f"progressive_{mode.lower()}",
            model=self.config.model,
            temperature=self.config.temperature,
            system_prompt=self.PROGRESSIVE_VLM_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            image_paths=raw_images[:frames_limit],
            response_text=text,
            parsed_response=parsed,
            clip_info=clip_info,
        )

        # 添加模式标记
        parsed["_vlm_mode"] = mode
        parsed["_n_frames"] = len(raw_images[:frames_limit])

        return parsed

    def analyze_progressive_two_stage(
        self,
        raw_images_fast: List[str],
        raw_images_escalated: List[str],
        metadata_text_fast: str,
        metadata_text_escalated: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
        escalate_on_verdicts: List[str] = None,
        escalate_on_conflict: bool = True,
        conflict_risk_threshold: float = 0.6,
        resolution_conservative: bool = True,
        # v2新增升级参数
        escalate_on_low_conf_no: bool = True,
        low_conf_no_threshold: float = 0.5,
        escalate_on_high_signal: bool = True,
        high_signal_threshold: float = 0.7,
        max_signal_score: float = 0.0,
    ) -> Dict:
        """
        两阶段渐进式VLM分析（v2支持更多升级规则）

        Args:
            raw_images_fast: S1快速验证帧
            raw_images_escalated: S2升级验证帧
            metadata_text_fast: S1元数据
            metadata_text_escalated: S2元数据
            user_query: 用户查询
            clip_info: clip信息
            escalate_on_verdicts: 触发升级的verdict列表
            escalate_on_conflict: 是否在risk高但verdict=NO时升级
            conflict_risk_threshold: conflict规则的risk阈值
            resolution_conservative: 使用保守解析规则
            escalate_on_low_conf_no: verdict=NO但置信度低时升级
            low_conf_no_threshold: 低置信度NO的阈值
            escalate_on_high_signal: 高信号分时升级
            high_signal_threshold: 高信号分阈值
            max_signal_score: 最大信号分（由keyframe_selector计算）

        Returns:
            包含两阶段结果的字典
        """
        if escalate_on_verdicts is None:
            escalate_on_verdicts = ["UNCERTAIN", "POST_EVENT_ONLY"]

        # ===== S1: 快速验证 =====
        _safe_print(f"\n[渐进式VLM] S1阶段：快速验证 ({len(raw_images_fast)}帧)")
        result_s1 = self.analyze_progressive(
            raw_images=raw_images_fast,
            metadata_text=metadata_text_fast,
            user_query=user_query,
            clip_info=clip_info,
            mode="FAST"
        )

        # 解析S1结果
        result_s1 = parse_vlm_response_with_verdict(result_s1)
        verdict_s1 = result_s1.get("verdict", "NO")
        conf_s1 = result_s1.get("confidence", 0.0)

        # 检查是否需要升级
        need_escalation = False
        escalation_reasons = []

        # 规则1: verdict在升级列表中
        if verdict_s1 in escalate_on_verdicts:
            need_escalation = True
            escalation_reasons.append(f"verdict={verdict_s1}")

        # 规则2: risk高但verdict=NO（冲突）
        if escalate_on_conflict and verdict_s1 == "NO":
            clip_score = clip_info.get("clip_score", 0) if clip_info else 0
            accident_score = clip_info.get("accident_score", 0) if clip_info else 0
            risk_score = max(clip_score, accident_score)

            if risk_score >= conflict_risk_threshold:
                need_escalation = True
                escalation_reasons.append(f"conflict: risk={risk_score:.2f}>={conflict_risk_threshold}")

        # 规则3 (v2): verdict=NO但置信度低
        if escalate_on_low_conf_no and verdict_s1 == "NO" and conf_s1 < low_conf_no_threshold:
            need_escalation = True
            escalation_reasons.append(f"low_conf_no: conf={conf_s1:.2f}<{low_conf_no_threshold}")

        # 规则4 (v2): 高信号分
        if escalate_on_high_signal and max_signal_score >= high_signal_threshold:
            need_escalation = True
            escalation_reasons.append(f"high_signal: {max_signal_score:.2f}>={high_signal_threshold}")

        escalation_reason = " | ".join(escalation_reasons) if escalation_reasons else ""

        # 如果不需要升级，直接返回S1结果
        if not need_escalation:
            _safe_print(f"[渐进式VLM] S1完成，无需升级 (verdict={verdict_s1}, conf={conf_s1:.2f})")
            return {
                "verdict": verdict_s1,
                "confidence": conf_s1,
                "has_accident": result_s1.get("has_accident", False),
                "accidents": result_s1.get("accidents", []),
                "text_summary": result_s1.get("text_summary", ""),
                "escalated": False,
                "stage1_result": result_s1,
                "stage2_result": None,
                "escalation_reason": "",
                "_progressive_vlm": True,
            }

        # ===== S2: 升级验证 =====
        _safe_print(f"\n[渐进式VLM] S2阶段：升级验证 ({len(raw_images_escalated)}帧)")
        _safe_print(f"[渐进式VLM] 升级原因: {escalation_reason}")

        result_s2 = self.analyze_progressive(
            raw_images=raw_images_escalated,
            metadata_text=metadata_text_escalated,
            user_query=user_query,
            clip_info=clip_info,
            mode="ESCALATED"
        )

        # 解析S2结果
        result_s2 = parse_vlm_response_with_verdict(result_s2)
        verdict_s2 = result_s2.get("verdict", "NO")
        conf_s2 = result_s2.get("confidence", 0.0)

        # ===== 结果解析 =====
        # 保守策略：优先避免FPR上升
        if resolution_conservative:
            # S2=YES → YES（信任S2）
            if verdict_s2 == "YES":
                final_verdict = "YES"
                final_conf = conf_s2
                final_has_accident = True
            # S2=POST_EVENT_ONLY 且高置信 → POST_EVENT_ONLY
            elif verdict_s2 == "POST_EVENT_ONLY" and conf_s2 >= 0.9:
                final_verdict = "POST_EVENT_ONLY"
                final_conf = conf_s2
                final_has_accident = True
            # S2=UNCERTAIN 且高置信 → UNCERTAIN
            elif verdict_s2 == "UNCERTAIN" and conf_s2 >= 0.9:
                final_verdict = "UNCERTAIN"
                final_conf = conf_s2
                final_has_accident = False
            # 其他情况保持S1结果（保守）
            else:
                final_verdict = verdict_s1
                final_conf = conf_s1
                final_has_accident = result_s1.get("has_accident", False)
        else:
            # 非保守策略：信任S2
            final_verdict = verdict_s2
            final_conf = conf_s2
            final_has_accident = result_s2.get("has_accident", False)

        _safe_print(f"[渐进式VLM] 解析结果: S1={verdict_s1}({conf_s1:.2f}) → S2={verdict_s2}({conf_s2:.2f}) → Final={final_verdict}")

        # 合并accidents
        accidents = result_s2.get("accidents", []) or result_s1.get("accidents", [])

        return {
            "verdict": final_verdict,
            "confidence": final_conf,
            "has_accident": final_has_accident,
            "accidents": accidents,
            "text_summary": result_s2.get("text_summary") or result_s1.get("text_summary", ""),
            "escalated": True,
            "stage1_result": result_s1,
            "stage2_result": result_s2,
            "escalation_reason": escalation_reason,
            "_progressive_vlm": True,
        }

    def _detect_difficult_scene(
        self,
        text_summary: str,
        stage3_config: Optional[dict] = None
    ) -> Tuple[bool, List[str]]:
        """检测困难场景（夜间/雨天/雾天/雪天）

        Args:
            text_summary: VLM返回的文本摘要
            stage3_config: Stage3配置（可选）

        Returns:
            (is_difficult, scene_conditions) 元组
        """
        if stage3_config is None:
            # 默认关键词
            night_kw = ["夜间", "夜晚", "夜色", "黑暗", "灯光", "车灯", "路灯", "光线不足", "low light"]
            rain_kw = ["雨天", "雨水", "下雨", "湿滑", "雨夜", "雨中", "rain", "wet"]
            snow_kw = ["雪天", "下雪", "积雪", "雪地", "冰雪", "snow"]
            fog_kw = ["雾天", "大雾", "浓雾", "能见度低", "fog", "visibility"]
        else:
            night_kw = stage3_config.get("night_keywords", [])
            rain_kw = stage3_config.get("rain_keywords", [])
            snow_kw = stage3_config.get("snow_keywords", [])
            fog_kw = stage3_config.get("fog_keywords", [])

        text_lower = text_summary.lower()
        conditions = []

        # 检测各类困难场景
        if any(kw in text_summary or kw.lower() in text_lower for kw in night_kw):
            conditions.append("夜间/低光照")
        if any(kw in text_summary or kw.lower() in text_lower for kw in rain_kw):
            conditions.append("雨天/湿滑路面")
        if any(kw in text_summary or kw.lower() in text_lower for kw in snow_kw):
            conditions.append("雪天/冰雪路面")
        if any(kw in text_summary or kw.lower() in text_lower for kw in fog_kw):
            conditions.append("雾天/低能见度")

        return len(conditions) > 0, conditions

    def analyze_progressive_three_stage(
        self,
        raw_images_fast: List[str],
        raw_images_escalated: List[str],
        raw_images_s3: List[str],
        metadata_text_fast: str,
        metadata_text_escalated: str,
        metadata_text_s3: str,
        user_query: str,
        clip_info: Optional[Dict] = None,
        stage3_config: Optional[dict] = None,
        # S1/S2参数（继承自两阶段）
        escalate_on_verdicts: List[str] = None,
        escalate_on_conflict: bool = True,
        conflict_risk_threshold: float = 0.6,
        resolution_conservative: bool = True,
        escalate_on_low_conf_no: bool = True,
        low_conf_no_threshold: float = 0.5,
        escalate_on_high_signal: bool = True,
        high_signal_threshold: float = 0.7,
        max_signal_score: float = 0.0,
    ) -> Dict:
        """三阶段渐进式VLM分析 - 支持S3困难场景增强

        当S2返回NO但检测到困难场景时，使用增强prompt进行S3分析。

        Args:
            raw_images_s3: S3阶段帧（与S2相同或更多）
            metadata_text_s3: S3元数据
            stage3_config: Stage3配置字典
            其他参数同 analyze_progressive_two_stage

        Returns:
            包含三阶段结果的字典
        """
        # 首先执行S1+S2
        result_two_stage = self.analyze_progressive_two_stage(
            raw_images_fast=raw_images_fast,
            raw_images_escalated=raw_images_escalated,
            metadata_text_fast=metadata_text_fast,
            metadata_text_escalated=metadata_text_escalated,
            user_query=user_query,
            clip_info=clip_info,
            escalate_on_verdicts=escalate_on_verdicts,
            escalate_on_conflict=escalate_on_conflict,
            conflict_risk_threshold=conflict_risk_threshold,
            resolution_conservative=resolution_conservative,
            escalate_on_low_conf_no=escalate_on_low_conf_no,
            low_conf_no_threshold=low_conf_no_threshold,
            escalate_on_high_signal=escalate_on_high_signal,
            high_signal_threshold=high_signal_threshold,
            max_signal_score=max_signal_score,
        )

        # 检查是否需要S3
        if stage3_config is None or not stage3_config.get("enabled", False):
            return result_two_stage

        verdict_s2 = result_two_stage.get("verdict", "NO")

        # S3触发条件检查
        need_s3 = False
        if stage3_config.get("trigger_on_s2_no", True) and verdict_s2 == "NO":
            need_s3 = True
        if stage3_config.get("trigger_on_s2_uncertain", True) and verdict_s2 == "UNCERTAIN":
            need_s3 = True

        if not need_s3:
            return result_two_stage

        # 检测困难场景
        text_summary = result_two_stage.get("text_summary", "")
        s2_result = result_two_stage.get("stage2_result", {})
        if s2_result:
            text_summary = s2_result.get("text_summary", "") or text_summary

        is_difficult, scene_conditions = self._detect_difficult_scene(text_summary, stage3_config)

        if not is_difficult:
            # 非困难场景，不触发S3
            return result_two_stage

        # ===== S3: 困难场景增强分析 =====
        _safe_print(f"\n[渐进式VLM] S3阶段：困难场景增强分析 ({len(raw_images_s3)}帧)")
        _safe_print(f"[渐进式VLM] 检测到困难场景: {', '.join(scene_conditions)}")

        # 构建增强prompt
        if stage3_config.get("prompt_injection_enabled", True):
            prompt_prefix = stage3_config.get("difficult_scene_prompt_prefix", "")
            scene_text = ", ".join(scene_conditions)
            enhanced_query = prompt_prefix.replace("{scene_conditions}", scene_text) + "\n" + user_query
        else:
            enhanced_query = user_query

        # 执行S3分析
        result_s3 = self.analyze_progressive(
            raw_images=raw_images_s3,
            metadata_text=metadata_text_s3,
            user_query=enhanced_query,
            clip_info=clip_info,
            mode="ESCALATED"  # S3使用ESCALATED模式帧数
        )

        # 解析S3结果
        result_s3 = parse_vlm_response_with_verdict(result_s3)
        verdict_s3 = result_s3.get("verdict", "NO")
        conf_s3 = result_s3.get("confidence", 0.0)

        _safe_print(f"[渐进式VLM] S3结果: verdict={verdict_s3}, conf={conf_s3:.2f}")

        # 确定最终结果
        # S3优先策略：S3的YES/UNCERTAIN/POST_EVENT_ONLY优先于S2的NO
        if verdict_s3 in ["YES", "POST_EVENT_ONLY"]:
            final_verdict = verdict_s3
            final_conf = conf_s3
            final_has_accident = True
        elif verdict_s3 == "UNCERTAIN":
            # 可选：将高置信度UNCERTAIN提升为YES
            if stage3_config.get("boost_uncertain_to_yes", False):
                threshold = stage3_config.get("uncertain_boost_threshold", 0.7)
                if conf_s3 >= threshold:
                    final_verdict = "YES"
                    final_conf = conf_s3
                    final_has_accident = True
                else:
                    final_verdict = "UNCERTAIN"
                    final_conf = conf_s3
                    final_has_accident = False
            else:
                final_verdict = "UNCERTAIN"
                final_conf = conf_s3
                final_has_accident = False
        else:
            # S3仍然返回NO，保持S2结果
            final_verdict = result_two_stage.get("verdict", "NO")
            final_conf = result_two_stage.get("confidence", 0.0)
            final_has_accident = result_two_stage.get("has_accident", False)

        _safe_print(f"[渐进式VLM] 三阶段最终: {verdict_s2} -> S3={verdict_s3} -> Final={final_verdict}")

        # 合并结果
        accidents = result_s3.get("accidents", []) or result_two_stage.get("accidents", [])

        return {
            "verdict": final_verdict,
            "confidence": final_conf,
            "has_accident": final_has_accident,
            "accidents": accidents,
            "text_summary": result_s3.get("text_summary") or result_two_stage.get("text_summary", ""),
            "escalated": result_two_stage.get("escalated", False),
            "stage3_used": True,
            "stage3_reason": f"difficult_scene: {', '.join(scene_conditions)}",
            "stage1_result": result_two_stage.get("stage1_result"),
            "stage2_result": result_two_stage.get("stage2_result"),
            "stage3_result": result_s3,
            "escalation_reason": result_two_stage.get("escalation_reason", ""),
            "_progressive_vlm": True,
            "_s3_n_frames": len(raw_images_s3),
            "s3_images_count": len(raw_images_s3),
        }

    def followup_chat(
        self,
        frame_paths: List[str],
        original_prompt: Optional[str],
        original_response: Optional[Dict],
        conversation_history: List[Dict],
        question: str
    ) -> str:
        """
        基于已完成的事故分析进行追问对话

        Args:
            frame_paths: 原分析使用的帧图片路径列表
            original_prompt: 原分析使用的prompt
            original_response: 原分析的sidecar响应（JSON）
            conversation_history: 之前的对话历史 [{"role": "user/assistant", "content": "..."}]
            question: 当前用户追问

        Returns:
            str: VLM的回答
        """
        _safe_print("\n" + "="*60)
        _safe_print("[VLM请求] 追问对话模式")
        _safe_print("="*60)
        _safe_print(f"[模型]: {self.config.model}")
        _safe_print(f"[问题]: {question[:100]}...")
        _safe_print(f"[历史轮数]: {len(conversation_history) // 2}")

        # 系统提示词
        system_prompt = """你是一个专业的交通事故分析专家。用户之前已经对一段视频进行了事故分析，现在想针对分析结果进行追问。

请注意：
1. 你可以看到原始分析使用的视频帧图像
2. 你可以看到之前的分析结果
3. 请基于这些视觉证据和分析结果回答用户的追问
4. 如果问题涉及帧中无法确定的内容，请诚实说明
5. 引用证据时，请具体指出是哪个帧、哪个位置

回答风格：
- 只返回纯文字描述，不要返回JSON格式
- 使用自然语言回答，像与人对话一样
- 简洁明了，直接回答问题
- 引用具体的帧号和视觉证据
- 如果分析有不确定性，说明置信度
"""

        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # 添加原始分析上下文（包含图片）
        original_content = []

        # 添加原始prompt
        if original_prompt:
            original_content.append({
                "type": "text",
                "text": f"[原始分析Prompt]\n{original_prompt[:2000]}..."  # 截断避免太长
            })

        # 添加帧图片（限制数量避免太多）
        max_frames = min(8, len(frame_paths))  # 最多显示8帧
        for i, path in enumerate(frame_paths[:max_frames]):
            if os.path.exists(path):
                try:
                    original_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_to_base64_url(path, max_width=800, quality=75)}
                    })
                except Exception as e:
                    _safe_print(f"[警告] 无法加载帧 {path}: {e}")

        if original_content:
            messages.append({"role": "user", "content": original_content})

        # 添加原始分析结果
        if original_response:
            # 将sidecar响应格式化为文本
            if isinstance(original_response, dict):
                response_text = json.dumps(original_response, ensure_ascii=False, indent=2)
            else:
                response_text = str(original_response)
            messages.append({
                "role": "assistant",
                "content": f"[事故分析结果]\n{response_text[:3000]}"  # 截断避免太长
            })

        # 添加对话历史
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # 添加当前追问
        messages.append({
            "role": "user",
            "content": question
        })

        # 调用VLM
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.7,  # 稍高一点以获得更自然的回答
                max_tokens=2048
            )

            answer = response.choices[0].message.content
            _safe_print(f"[VLM响应] {answer[:200]}...")
            _safe_print("="*60 + "\n")

            return answer

        except Exception as e:
            _safe_print(f"[VLM错误] {str(e)}")
            raise RuntimeError(f"VLM调用失败: {str(e)}")
