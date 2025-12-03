from __future__ import annotations

from typing import Dict, List, Tuple

from .config import TemplateConfig


VIOLATION_KEYWORDS = {
    "bike_wrong_way": ["逆行", "wrong way", "逆向"],
    "run_red_light": ["闯红灯", "run red", "红灯"],
    "occupy_motor_lane": ["占用机动车道", "机动", "占道", "机动车道"],
    "accident": ["事故", "碰撞", "摔倒", "accident", "collision"],
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
