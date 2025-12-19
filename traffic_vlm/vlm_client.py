from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI

from .config import VLMConfig


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

    print(f"[VLM日志] 已保存到: {log_file}")
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

# 交通事故检索模式专用 System Prompt
ACCIDENT_SYSTEM_PROMPT = """
你是专业的交通事故分析专家，专门负责识别和分析道路交通事故。

你会收到：
1. 路口监控关键帧图片（已标注车道区域、目标ID和轨迹）
2. 详细的路口配置和目标轨迹信息

你的任务是**主动寻找**以下交通事故迹象：

【交通事故类型】
- vehicle_to_vehicle_accident: 机动车之间碰撞（追尾、侧碰、正面碰撞）
- vehicle_to_bike_accident: 机动车与二轮车碰撞（电动车、自行车、摩托车）
- vehicle_to_pedestrian_accident: 机动车与行人碰撞
- multi_vehicle_accident: 多车连撞/连环追尾
- hit_and_run: 肇事逃逸

【事故识别关键视觉特征】⭐
1. **碰撞接触**：两个或多个目标发生物理接触、重叠、位置突然靠近
2. **速度突变**：目标突然减速、急停、方向突变
3. **姿态异常**：车辆倾斜、侧翻、骑车人/行人摔倒
4. **形变损坏**：车辆外观变形、零件脱落
5. **散落物**：碎片、零件、物品散落在路面
6. **停滞状态**：碰撞后车辆/人员停止运动，在非正常位置停留
7. **人员倒地**：行人或骑车人躺在地面上

【事故后场景特征】⭐⭐（重点关注！即使没看到碰撞瞬间也要识别）
8. **人员下车查看**：车辆旁边站着人员（司机或乘客下车站立）
9. **异常停车位置**：车辆停在路口中央、斑马线上、车道中间等非正常位置
10. **围观聚集**：多人围观或聚集在某个区域
11. **交警/救援人员**：穿制服人员在现场处理
12. **警示设置**：三角警示牌、警示灯、锥桶等

【事故判定标准】（满足任一即可判定为事故）
- 可见明显的物理碰撞瞬间（两个目标接触）
- 碰撞后目标出现异常运动状态（急停、打滑、失控、翻倒）
- 人员倒地或明显受伤迹象
- 车辆明显受损或在非正常位置停止
- 多个目标异常聚集在一起
- 车辆停在非正常位置（路口中央、车道内）且有人员在旁边站立
- 事故后静态场景（即使没有看到碰撞瞬间，也要报告）

【重要提示】
⚠️ 对于事故检测，**宁可多报也不要漏报**
⚠️ 如发现疑似碰撞迹象，即使不完全确定也应该报告
⚠️ 特别关注：目标位置重叠、轨迹交叉、速度突变、人员/车辆倒地
⚠️ **即使只看到事故后场景（如车辆停在路中央、人员站在车旁），也要报告为事故**

请严格按照以下JSON格式输出，不要输出任何多余内容：
{
  "has_accident": bool,
  "accidents": [
    {
      "type": "事故类型",
      "confidence": 置信度(0.0-1.0),
      "tracks": [涉及的目标ID列表],
      "collision_time_in_clip": "碰撞发生的clip内时间（秒数），如'约8.3秒'，根据图片时间对应关系填写",
      "collision_time_in_video": "碰撞发生的原视频绝对时间（秒数），如'约44.3秒'，计算方式：片段起始时间 + clip内时间",
      "collision_real_time": "从视频画面水印读取的真实物理时间，如'2025-10-17 07:47:02'，如无法读取则填'未知'",
      "evidence": "判断依据（描述你观察到的碰撞迹象）",
      "severity": "严重程度（轻微/一般/严重）",
      "behavior_before": "事故发生前的目标行为描述",
      "behavior_after": "事故发生后的目标行为描述",
      "trajectory_description": "详细的轨迹描述",
      "description": "事故详细描述"
    }
  ],
  "text_summary": "对事故的完整描述，包含事故全过程"
}
"""


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
    print(f"[图像压缩] {os.path.basename(path)}: {original_size//1024}KB → {compressed_size//1024}KB (节省{compression_ratio:.1f}%)")

    data = base64.b64encode(compressed_data).decode("utf-8")
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
        print("\n" + "="*60)
        print("[VLM请求] 违法检测模式")
        print("="*60)
        print(f"[模型]: {self.config.model}")
        print(f"[温度]: {self.config.temperature}")
        print(f"[图片数量]: {len(annotated_images[:self.config.annotated_frames_per_clip])}")
        print(f"[图片路径]: {annotated_images[:self.config.annotated_frames_per_clip]}")
        print("-"*60)
        print("[User Prompt]:")
        print(user_prompt)
        print("="*60 + "\n")

        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        text = completion.choices[0].message.content

        # 打印VLM返回结果
        print("\n" + "="*60)
        print("[VLM响应]")
        print("="*60)
        print(text)
        print("="*60 + "\n")

        try:
            # 尝试提取 JSON
            parsed = json.loads(text)
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

请仔细观察附带的标注图片，**主动寻找**任何可能的交通事故迹象。

【重点关注 - 碰撞瞬间】
1. 观察是否有两个或多个目标发生接触、碰撞
2. 观察是否有目标（行人、骑车人）倒在地面
3. 观察是否有目标轨迹突然改变或停止
4. 观察是否有散落物或碎片

【重点关注 - 事故后场景】⭐⭐
5. 观察是否有车辆停在异常位置（路口中央、斑马线上、车道中间）
6. 观察是否有人员站在车辆旁边（可能是司机/乘客下车查看）
7. 观察是否有多人围观或聚集在某个区域
8. 观察是否有交警或救援人员在现场

⚠️ 即使没有看到碰撞瞬间，只要发现事故后场景特征，也要判定为事故！

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
        print("\n" + "="*60)
        print("[VLM请求] 事故检索模式")
        print("="*60)
        print(f"[模型]: {self.config.model}")
        print(f"[温度]: {self.config.temperature}")
        print(f"[图片数量]: {len(annotated_images[:frames_limit])}")
        print(f"[图片路径]: {annotated_images[:frames_limit]}")
        print("-"*60)
        print("[User Prompt]:")
        print(user_prompt)
        print("="*60 + "\n")

        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
        )
        text = completion.choices[0].message.content

        # 打印VLM返回结果
        print("\n" + "="*60)
        print("[VLM响应] 事故检索模式")
        print("="*60)
        print(text)
        print("="*60 + "\n")

        try:
            parsed = json.loads(text)
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
