"""
事故复盘描述 Sidecar 模块

在不改变现有事故判别pipeline的前提下，增加一个"事故复盘描述 sidecar"：
- 默认关闭，不影响任何现有行为
- 开启后只写附加文件，不影响最终判别结果
- 仅当 verdict 命中触发条件时，额外调用多个VLM模型生成JSON报告
- 固定解码参数保证可复现
- 异常不影响主流程
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from traffic_vlm.config import NarrativeSidecarConfig
from traffic_vlm.vlm_client import image_to_base64_url

logger = logging.getLogger(__name__)


def _load_prompt_template(config: NarrativeSidecarConfig) -> str:
    """加载 prompt 模板文件"""
    # 获取模块所在目录
    module_dir = Path(__file__).parent
    template_path = module_dir / config.prompt_template_file

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt模板文件不存在: {template_path}")

    return template_path.read_text(encoding="utf-8")


def _extract_json_from_text(text: str) -> Optional[Dict]:
    """从VLM响应中提取JSON"""
    text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None


def _call_single_model(
    model_id: str,
    prompt: str,
    frames: List[str],
    config: NarrativeSidecarConfig,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    """调用单个VLM模型"""
    start_time = time.time()
    result = {
        "model_id": model_id,
        "parse_ok": False,
        "accident_status": None,
        "confidence": None,
        "latency_ms": 0,
        "unknowns_count": 0,
        "quality_warnings_count": 0,
        "has_impact_moment": False,
        "error": None,
        "raw_text": "",
        "parsed_response": None,
    }

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)

        # 构建消息内容
        contents: List[Dict] = [{"type": "text", "text": prompt}]

        # 添加图像
        for img_path in frames:
            if os.path.exists(img_path):
                contents.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_base64_url(img_path)},
                })

        messages = [{"role": "user", "content": contents}]

        # 调用API
        api_params = {
            "model": model_id,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }

        completion = client.chat.completions.create(**api_params)
        raw_text = completion.choices[0].message.content or ""
        result["raw_text"] = raw_text

        # 解析JSON
        parsed = _extract_json_from_text(raw_text)
        if parsed:
            result["parse_ok"] = True
            result["parsed_response"] = parsed
            result["accident_status"] = parsed.get("accident_status")
            result["confidence"] = parsed.get("confidence")
            result["unknowns_count"] = len(parsed.get("unknowns", []))
            result["quality_warnings_count"] = len(parsed.get("quality_warnings", []))

            # 检查是否有 impact_moment
            collision = parsed.get("collision_assessment", {})
            impact = collision.get("impact_moment", {})
            result["has_impact_moment"] = bool(impact.get("estimated_time"))

    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"Sidecar model {model_id} call failed: {e}")

    result["latency_ms"] = int((time.time() - start_time) * 1000)
    return result


def run_narrative_sidecar(
    *,
    casebook_dir: Path,
    video_name: str,
    final_verdict: str,
    final_confidence: float,
    frames: List[str],
    frame_timestamps_sec: List[float],
    clip_meta: Dict,
    config: NarrativeSidecarConfig,
    custom_prompt: Optional[str] = None,
    yolo_tracks: Optional[str] = None,
) -> Dict:
    """
    执行事故复盘描述 sidecar

    Args:
        casebook_dir: casebook输出目录
        video_name: 视频文件名
        final_verdict: 主流程最终verdict (YES/NO/UNCERTAIN/POST_EVENT_ONLY)
        final_confidence: 主流程最终置信度
        frames: 帧路径列表（与主流程最后一次VLM输入相同）
        frame_timestamps_sec: 帧时间戳列表（秒）
        clip_meta: clip元数据（包含start_time, end_time等）
        config: sidecar配置
        custom_prompt: 用户自定义prompt模板（可选），如果提供则替代默认模板
        yolo_tracks: YOLO检测轨迹的文本描述（可选），会替换模板中的{yolo_tracks}占位符

    Returns:
        {
            "enabled": bool,
            "triggered": bool,
            "models": {...},
            "out_dir": str,
            "error": str (optional)
        }
    """
    sidecar_result = {
        "enabled": True,
        "triggered": False,
        "models": {},
        "out_dir": config.out_dirname,
    }

    try:
        # 1. 检查触发条件
        trigger_verdicts = []
        if config.trigger_on_yes:
            trigger_verdicts.append("YES")
        if config.trigger_on_uncertain:
            trigger_verdicts.append("UNCERTAIN")
        if config.trigger_on_post_event_only:
            trigger_verdicts.append("POST_EVENT_ONLY")

        if final_verdict not in trigger_verdicts:
            logger.debug(f"Sidecar not triggered: verdict={final_verdict}, triggers={trigger_verdicts}")
            return sidecar_result

        sidecar_result["triggered"] = True

        # 2. 创建输出目录
        out_dir = Path(casebook_dir) / config.out_dirname
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2.5 检查是否已有缓存结果，跳过重复处理
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            try:
                cached = json.loads(summary_path.read_text(encoding="utf-8"))
                models_data = cached.get("models", {})

                # 补充加载每个模型的response.json（兼容旧缓存）
                for model_id in models_data:
                    if not models_data[model_id].get('parsed_response'):
                        safe_model_id = model_id.replace('/', '_')
                        response_path = out_dir / safe_model_id / 'response.json'
                        if response_path.exists():
                            try:
                                models_data[model_id]['parsed_response'] = json.loads(
                                    response_path.read_text(encoding='utf-8')
                                )
                                logger.debug(f"Loaded parsed_response from {response_path}")
                            except Exception as e:
                                logger.warning(f"Failed to load response.json for {model_id}: {e}")

                sidecar_result["models"] = models_data
                sidecar_result["cached"] = True
                logger.info(f"Sidecar skipped (cached): {video_name}")
                return sidecar_result
            except Exception:
                pass  # 缓存损坏，继续重新处理

        # 3. 加载并构造 prompt（支持自定义prompt）
        if custom_prompt:
            prompt_template = custom_prompt
        else:
            prompt_template = _load_prompt_template(config)

        # 格式化时间戳
        frame_times_str = json.dumps(frame_timestamps_sec) if frame_timestamps_sec else "[]"

        # 使用 str.replace() 而非 .format()，避免 JSON Schema 中的 {} 被误解析
        prompt = prompt_template
        prompt = prompt.replace("{video_name}", str(video_name))
        prompt = prompt.replace("{clip_start_sec}", str(clip_meta.get("start_time", "UNKNOWN")))
        prompt = prompt.replace("{clip_end_sec}", str(clip_meta.get("end_time", "UNKNOWN")))
        prompt = prompt.replace("{fps_or_frame_times}", frame_times_str)
        prompt = prompt.replace("{model_verdict}", str(final_verdict))
        prompt = prompt.replace("{model_confidence}", str(final_confidence))
        prompt = prompt.replace("{yolo_tracks}", yolo_tracks or "无YOLO轨迹数据")

        # 4. 获取API凭据
        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY 环境变量")

        # 5. 并行调用所有模型
        model_results = {}

        with ThreadPoolExecutor(max_workers=config.parallelism) as executor:
            futures = {
                executor.submit(
                    _call_single_model,
                    model_id,
                    prompt,
                    frames,
                    config,
                    api_key,
                    base_url,
                ): model_id
                for model_id in config.models
            }

            for future in as_completed(futures):
                model_id = futures[future]
                try:
                    result = future.result(timeout=config.timeout_sec)
                    model_results[model_id] = result
                except Exception as e:
                    model_results[model_id] = {
                        "model_id": model_id,
                        "parse_ok": False,
                        "error": str(e),
                        "latency_ms": 0,
                    }

        # 6. 写入每个模型的输出
        for model_id, result in model_results.items():
            model_dir = out_dir / model_id.replace("/", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            # response.json - 解析后的JSON
            if result.get("parsed_response"):
                (model_dir / "response.json").write_text(
                    json.dumps(result["parsed_response"], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )

            # raw.txt - 原始响应
            if result.get("raw_text"):
                (model_dir / "raw.txt").write_text(result["raw_text"], encoding="utf-8")

            # request_meta.json - 请求元数据
            request_meta = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "video_name": video_name,
                "final_verdict": final_verdict,
                "final_confidence": final_confidence,
                "frame_count": len(frames),
                "frame_timestamps_range": [
                    min(frame_timestamps_sec) if frame_timestamps_sec else 0,
                    max(frame_timestamps_sec) if frame_timestamps_sec else 0,
                ],
                "latency_ms": result.get("latency_ms", 0),
                "parse_ok": result.get("parse_ok", False),
                "error": result.get("error"),
                "config": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "max_tokens": config.max_tokens,
                },
            }
            (model_dir / "request_meta.json").write_text(
                json.dumps(request_meta, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            # 更新汇总结果
            sidecar_result["models"][model_id] = {
                "parse_ok": result.get("parse_ok", False),
                "accident_status": result.get("accident_status"),
                "confidence": result.get("confidence"),
                "latency_ms": result.get("latency_ms", 0),
                "unknowns_count": result.get("unknowns_count", 0),
                "quality_warnings_count": result.get("quality_warnings_count", 0),
                "has_impact_moment": result.get("has_impact_moment", False),
                "error": result.get("error"),
                # 新增：保存完整的解析结果供前端展示
                "parsed_response": result.get("parsed_response"),
            }

        # 7. 生成聚合文件 summary.json
        summary = {
            "video_name": video_name,
            "trigger_verdict": final_verdict,
            "trigger_confidence": final_confidence,
            "timestamp": datetime.now().isoformat(),
            "frame_count": len(frames),
            "models": sidecar_result["models"],
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        logger.info(f"Sidecar completed for {video_name}: {len(model_results)} models, "
                    f"parse_ok={sum(1 for m in model_results.values() if m.get('parse_ok'))}")

    except Exception as e:
        # 异常处理：写入错误日志，不抛出
        error_msg = str(e)
        sidecar_result["error"] = error_msg
        logger.warning(f"Sidecar failed (non-fatal) for {video_name}: {e}")

        try:
            error_dir = Path(casebook_dir) / config.out_dirname
            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / "sidecar_error.json"
            error_path.write_text(json.dumps({
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
                "video_name": video_name,
                "final_verdict": final_verdict,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass  # 错误日志写入失败也不抛出

    return sidecar_result
