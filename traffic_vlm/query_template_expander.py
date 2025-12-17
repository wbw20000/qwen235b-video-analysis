from __future__ import annotations

from typing import Dict, List, Tuple

from .config import TemplateConfig


VIOLATION_KEYWORDS = {
    # 二轮车违法行为
    "bike_wrong_way": ["逆行", "wrong way", "逆向", "二轮车逆行", "电动车逆行", "自行车逆行"],
    "run_red_light_bike": ["二轮车闯红灯", "电动车闯红灯", "自行车闯红灯", "二轮车红灯"],
    "occupy_motor_lane_bike": ["二轮车占用机动车道", "电动车占道", "自行车占道", "非机动车占道"],
    "bike_improper_turning": ["二轮车违规转弯", "电动车违规变道", "自行车违规转向"],
    "bike_illegal_u_turn": ["二轮车违规掉头", "电动车掉头", "自行车掉头"],

    # 机动车违法行为
    "car_wrong_way": ["机动车逆行", "汽车逆行", "车辆逆行", "小车逆行"],
    "run_red_light_car": ["机动车闯红灯", "汽车闯红灯", "车辆闯红灯", "小车红灯"],
    "illegal_parking": ["违法停车", "违停", "乱停车", "禁停区域停车"],
    "illegal_u_turn": ["机动车违规掉头", "汽车掉头", "车辆掉头", "小车掉头"],
    "speeding": ["超速", "车速过快", "超过限速"],
    "illegal_overtaking": ["违规超车", "禁止超车", "不当超车"],
    "improper_lane_change": ["违规变道", "不当变道", "突然变道", "未打转向灯变道"],

    # 交通事故
    "vehicle_to_vehicle_accident": ["两车相撞", "车车事故", "机动车碰撞", "汽车相撞", "车辆碰撞"],
    "vehicle_to_bike_accident": ["车与二轮车事故", "车撞电动车", "车撞自行车", "机动车与二轮车碰撞", "汽车撞电动车"],
    "vehicle_to_pedestrian_accident": ["车撞人", "机动车撞行人", "汽车撞人", "车辆撞人"],
    "multi_vehicle_accident": ["多车连撞", "连环撞车", "三车相撞", "多车事故"],
    "hit_and_run": ["肇事逃逸", "逃逸", "撞车后逃跑", "事故逃逸"],
}


# 事故检索模式专用关键词（更宽松的匹配）
ACCIDENT_KEYWORDS = {
    "vehicle_to_vehicle_accident": [
        "事故", "交通事故", "车祸", "碰撞", "撞车", "相撞", "追尾",
        "两车相撞", "车车事故", "机动车碰撞", "汽车相撞", "车辆碰撞",
        "accident", "collision", "crash",
    ],
    "vehicle_to_bike_accident": [
        "撞电动车", "撞自行车", "撞摩托车", "车撞电动车", "车撞自行车",
        "机动车与二轮车碰撞", "汽车撞电动车", "车与二轮车事故",
    ],
    "vehicle_to_pedestrian_accident": [
        "撞人", "撞行人", "车撞人", "机动车撞行人", "汽车撞人", "车辆撞人",
    ],
    "multi_vehicle_accident": [
        "多车", "连环", "三车", "多车连撞", "连环撞车", "多车事故",
    ],
    "hit_and_run": [
        "逃逸", "肇事逃逸", "撞车后逃跑", "事故逃逸",
    ],
}


def infer_violation_types(user_query: str) -> List[str]:
    """根据用户查询粗略推断违规类型列表。"""
    matched = []
    for v_type, keywords in VIOLATION_KEYWORDS.items():
        if any(k.lower() in user_query.lower() for k in keywords):
            matched.append(v_type)
    if not matched:
        matched.append("bike_wrong_way")
    return matched


def infer_accident_types(user_query: str) -> List[str]:
    """
    事故检索模式：更积极地匹配所有事故类型。
    如果没有匹配或输入包含通用"事故"关键词，返回所有事故类型。
    """
    matched = []
    for v_type, keywords in ACCIDENT_KEYWORDS.items():
        if any(k.lower() in user_query.lower() for k in keywords):
            matched.append(v_type)

    # 如果没有匹配或输入包含通用"事故"词，返回所有事故类型
    generic_accident_terms = ["事故", "交通事故", "accident", "碰撞", "撞车"]
    if not matched or any(term in user_query.lower() for term in generic_accident_terms):
        return list(ACCIDENT_KEYWORDS.keys())
    return matched


def expand_templates(user_query: str, template_config: TemplateConfig) -> Tuple[List[str], List[str]]:
    """
    用户 query -> 模板列表。
    返回 (templates, violation_types)
    """
    violation_types = infer_violation_types(user_query)
    templates: List[str] = []

    for v_type in violation_types:
        templates.extend(template_config.builtin_templates.get(v_type, []))

    # 用户原始 query 也加入模板，提升专用性
    templates.append(user_query)
    # 去重
    templates = list(dict.fromkeys(templates))
    return templates, violation_types


def expand_accident_templates(user_query: str, template_config: TemplateConfig) -> Tuple[List[str], List[str]]:
    """
    事故检索模式：扩展事故相关模板。
    返回 (templates, accident_types)
    """
    accident_types = infer_accident_types(user_query)
    templates: List[str] = []

    for v_type in accident_types:
        templates.extend(template_config.builtin_templates.get(v_type, []))

    # 用户原始 query 也加入模板
    templates.append(user_query)
    # 去重
    templates = list(dict.fromkeys(templates))
    return templates, accident_types
