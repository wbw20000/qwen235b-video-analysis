#!/usr/bin/env python3
"""
test_3d_grounding.py - 测试 Qwen3-VL-32B-AWQ 的 grounding 能力与 YOLO 对比

功能:
  1. 读取 cache/grounding_test_input.json (Team A 生成的 S2 帧 + YOLO 检测)
  2. 将多帧图片发送到远程 vLLM，要求 VLM 做目标检测 + 跨帧 ID + 深度估算
  3. 与 YOLO/ByteTrack 结果做 IoU 对比
  4. 在原图上画出 VLM 框(绿色) 和 YOLO 框(红色虚线)
  5. 生成 4x3 网格总览图 + 对比报告 + 时间线

使用:
  # 完整流程（调用 VLM）
  d:/project2025/qwen235b/venv/Scripts/python.exe test_3d_grounding.py

  # 从缓存重新生成可视化（跳过 VLM 调用）
  d:/project2025/qwen235b/venv/Scripts/python.exe test_3d_grounding.py --from-cache
"""

import json
import os
import sys
import base64
import re
import math
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BASE_DIR = Path("d:/project2025/qwen235b")
INPUT_FILE = BASE_DIR / "cache" / "grounding_test_input.json"
OUTPUT_DIR = BASE_DIR / "outputs"
VLLM_BASE_URL = "http://100.123.59.56:8000/v1"
VLLM_MODEL = "qwen3-vl-32b"
VLLM_API_KEY = "EMPTY"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def image_to_base64_url(path: str, max_width: int = 640, quality: int = 85) -> str:
    """将图像文件转换为 base64 data URL，缩放到 max_width 以控制 token 数量。

    4K 原图 → 640px: 视觉 token 从 ~270/图 降到 ~70/图，12帧 prefill 从15min降到2min。
    """
    from PIL import Image as _PILImage
    import io as _io

    with _PILImage.open(path) as img:
        img = img.convert("RGB")
        W, H = img.size
        if max_width and W > max_width:
            scale = max_width / W
            new_w = max_width
            new_h = int(H * scale)
            img = img.resize((new_w, new_h), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def resolve_frame_image_path(frame_info: dict) -> Optional[Path]:
    """按优先级解析帧图片的绝对路径。优先 raw_path，其次 annotated_path。"""
    for key in ("raw_path", "annotated_path"):
        rel = frame_info.get(key, "")
        if not rel:
            continue
        # 支持绝对路径和相对路径
        p = Path(rel)
        if p.is_absolute() and p.exists():
            return p
        p = BASE_DIR / rel
        if p.exists():
            return p
    return None


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个框的 IoU。
    box 格式: [x1, y1, x2, y2]，归一化坐标 (0~1)。
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area < 1e-8:
        return 0.0
    return inter_area / union_area


def normalize_yolo_bbox(bbox_xyxy: List[float], width: int, height: int) -> List[float]:
    """将 YOLO 绝对像素坐标 [x1,y1,x2,y2] 转为归一化 [0~1]。"""
    return [
        bbox_xyxy[0] / width,
        bbox_xyxy[1] / height,
        bbox_xyxy[2] / width,
        bbox_xyxy[3] / height,
    ]


def parse_vlm_json(text: str) -> Optional[dict]:
    """
    健壮地解析 VLM 返回的 JSON。
    1. 先直接解析
    2. 失败则剥离 markdown 代码块后重试
    3. 再失败则用正则提取最大的 {...} 块
    """
    # 第一次直接解析
    text_stripped = text.strip()
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass

    # 剥离 markdown 代码块
    md_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    md_match = md_pattern.search(text_stripped)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 正则提取最大的 {...} 块
    brace_depth = 0
    start = None
    best_start = None
    best_end = None
    best_len = 0
    for i, ch in enumerate(text_stripped):
        if ch == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                length = i - start + 1
                if length > best_len:
                    best_start = start
                    best_end = i + 1
                    best_len = length
                start = None

    if best_start is not None:
        try:
            return json.loads(text_stripped[best_start:best_end])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# VLM 调用
# ---------------------------------------------------------------------------

def build_grounding_prompt(frames: List[dict], img_w: int = 0, img_h: int = 0) -> Tuple[str, str]:
    """构造 grounding prompt 的 system 和 user 文本。

    img_w, img_h: 图片实际像素尺寸（用于指导模型输出像素坐标范围）。
    """
    n = len(frames)
    # 估算帧间隔
    if n >= 2:
        interval = (frames[-1]["timestamp"] - frames[0]["timestamp"]) / max(n - 1, 1)
    else:
        interval = 0.0

    # 坐标范围说明
    if img_w > 0 and img_h > 0:
        coord_desc = (
            f"输出像素坐标边界框 [x1, y1, x2, y2]，"
            f"x 范围 0~{img_w}，y 范围 0~{img_h}，左上角为原点"
        )
        bbox_example = f"[{int(img_w*0.12)}, {int(img_h*0.45)}, {int(img_w*0.35)}, {int(img_h*0.72)}]"
    else:
        coord_desc = "输出归一化边界框 [x1, y1, x2, y2]（0.0~1.0，左上角为原点）"
        bbox_example = "[0.12, 0.45, 0.35, 0.72]"

    system_prompt = (
        "你是交通监控视频分析专家。请对给定的交通监控图像序列进行目标检测和跟踪分析。\n"
        "输出格式必须是严格的 JSON，不要有任何其他文字。"
    )

    user_prompt = (
        f"以下是来自同一路口的 {n} 张连续交通监控帧"
        f"（帧序号 0 到 {n-1}，间隔约 {interval:.1f} 秒）。\n"
    )
    if img_w > 0 and img_h > 0:
        user_prompt += f"图片分辨率：{img_w}x{img_h} 像素。\n"
    user_prompt += (
        "\n请对每一帧检测所有车辆（小轿车/货车/公交/摩托车/电动车）和行人：\n"
        f"1. {coord_desc}\n"
        "2. 跨帧目标ID必须连续稳定：同一物理目标在所有帧中使用同一个整数ID（从1开始）。"
        "若目标暂时消失后重现，保留原ID。严禁ID跳变。\n"
        "3. 估算每个目标距摄像头的深度：近(<20m) / 中(20-50m) / 远(>50m)，"
        "并给出估算距离（米）。\n\n"
        "输出纯JSON格式（无markdown代码块）：\n"
        "{\n"
        '  "frames": [\n'
        "    {\n"
        '      "frame_idx": 0,\n'
        '      "timestamp": 0.0,\n'
        '      "objects": [\n'
        f'        {{"id": 1, "category": "car", "bbox": {bbox_example}, '
        '"depth_zone": "近", "depth_m": 12}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "track_summary": {\n'
        '    "1": {"category": "car", "frames_present": [0,1,2,3], '
        '"continuous": true, "note": ""}\n'
        "  }\n"
        "}"
    )
    return system_prompt, user_prompt


def call_vlm_grounding(frames: List[dict]) -> Optional[dict]:
    """
    将多帧图片发送给 VLM，获取 grounding JSON。
    带重试逻辑。
    """
    from openai import OpenAI
    from PIL import Image as _PILImage

    # timeout=1200s：640px × 12帧 prefill ~2min + 生成 ~7min，留余量
    client = OpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL, timeout=1200.0)

    # 计算发送给 VLM 的实际图片尺寸（缩放后），写入 prompt 坐标范围
    MAX_SEND_WIDTH = 640
    img_w, img_h = 0, 0
    for frame in frames:
        fp = resolve_frame_image_path(frame)
        if fp is not None:
            try:
                with _PILImage.open(str(fp)) as _img:
                    orig_w, orig_h = _img.size
                if orig_w > MAX_SEND_WIDTH:
                    img_w = MAX_SEND_WIDTH
                    img_h = int(orig_h * MAX_SEND_WIDTH / orig_w)
                else:
                    img_w, img_h = orig_w, orig_h
                print(f"  原始图片: {orig_w}x{orig_h} → 发送尺寸: {img_w}x{img_h} px")
            except Exception:
                pass
            break

    system_prompt, user_prompt = build_grounding_prompt(frames, img_w, img_h)

    # 构建消息体：system + user(文本 + 多张图片)
    content_parts: List[dict] = [{"type": "text", "text": user_prompt}]

    for i, frame in enumerate(frames):
        img_path = resolve_frame_image_path(frame)
        if img_path is None:
            print(f"  [警告] 帧 {frame.get('frame_idx', i)} 图片不存在，跳过")
            continue
        print(f"  编码帧 {frame.get('frame_idx', i)}: {img_path.name} "
              f"({img_path.stat().st_size // 1024} KB)")
        b64_url = image_to_base64_url(str(img_path))
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": b64_url},
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  VLM 请求 (尝试 {attempt}/{MAX_RETRIES}) ...")
            t0 = time.time()
            response = client.chat.completions.create(
                model=VLLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
            )
            elapsed = time.time() - t0
            raw_text = response.choices[0].message.content or ""
            print(f"  VLM 响应: {len(raw_text)} 字符, 耗时 {elapsed:.1f}s")

            parsed = parse_vlm_json(raw_text)
            if parsed is not None:
                print(f"  JSON 解析成功: {len(parsed.get('frames', []))} 帧")
                return parsed
            else:
                print(f"  [警告] JSON 解析失败，原始文本前500字符:")
                print(f"  {raw_text[:500]}")
                if attempt < MAX_RETRIES:
                    print(f"  等待 {RETRY_DELAY}s 后重试 ...")
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  [错误] VLM 调用异常: {e}")
            if attempt < MAX_RETRIES:
                print(f"  等待 {RETRY_DELAY}s 后重试 ...")
                time.sleep(RETRY_DELAY)
            else:
                traceback.print_exc()

    print("  [失败] VLM 调用全部重试失败")
    return None


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _get_font(size: int):
    """加载 TrueType 字体，失败则 fallback 到默认字体。"""
    from PIL import ImageFont

    # 尝试常见字体路径
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/arial.ttf",      # Arial
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    # fallback
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _draw_dashed_rect(draw, box_px, color, width, dash_len=10, gap_len=6):
    """在 PIL ImageDraw 上画虚线矩形。"""
    x1, y1, x2, y2 = box_px

    def _dashed_line(start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        drawing = True
        while pos < length:
            seg_len = dash_len if drawing else gap_len
            seg_end = min(pos + seg_len, length)
            if drawing:
                sx = start[0] + ux * pos
                sy = start[1] + uy * pos
                ex = start[0] + ux * seg_end
                ey = start[1] + uy * seg_end
                draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            pos = seg_end
            drawing = not drawing

    _dashed_line((x1, y1), (x2, y1))  # top
    _dashed_line((x2, y1), (x2, y2))  # right
    _dashed_line((x2, y2), (x1, y2))  # bottom
    _dashed_line((x1, y2), (x1, y1))  # left


def draw_frame_annotations(
    img_path: Path,
    frame_idx: int,
    timestamp: float,
    vlm_objects: List[dict],
    yolo_objects: List[dict],
    video_width: int,
    video_height: int,
    output_path: Path,
) -> Path:
    """
    在原图上叠画 VLM 框(绿色) 和 YOLO 框(红色虚线)，保存到 output_path。
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(str(img_path)).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    font_vlm = _get_font(18)
    font_yolo = _get_font(16)
    font_info = _get_font(20)

    # --- 左上角信息栏 ---
    info_text = f"帧{frame_idx} t={timestamp:.2f}s"
    # 使用 textbbox 获取文本大小
    try:
        tbbox = draw.textbbox((0, 0), info_text, font=font_info)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
    except AttributeError:
        tw, th = draw.textsize(info_text, font=font_info)
    padding = 6
    draw.rectangle(
        [0, 0, tw + padding * 2, th + padding * 2],
        fill=(0, 0, 0, 160),
    )
    draw.text((padding, padding), info_text, fill=(255, 255, 255, 255), font=font_info)

    # --- YOLO 框 (红色虚线) ---
    yolo_color = (220, 50, 50)
    for det in yolo_objects:
        bbox = det.get("bbox_xyxy", [])
        if len(bbox) != 4:
            continue
        # 像素坐标到图片坐标（图片尺寸可能与 video_width/height 不同）
        scale_x = W / video_width
        scale_y = H / video_height
        px1 = bbox[0] * scale_x
        py1 = bbox[1] * scale_y
        px2 = bbox[2] * scale_x
        py2 = bbox[3] * scale_y

        _draw_dashed_rect(draw, (px1, py1, px2, py2), yolo_color, width=2)

        # 标签
        tid = det.get("track_id", "?")
        cat = det.get("category", "?")
        score = det.get("score", 0)
        label = f"YOLO#{tid} {cat} {score:.2f}"
        try:
            lbbox = draw.textbbox((0, 0), label, font=font_yolo)
            lw = lbbox[2] - lbbox[0]
            lh = lbbox[3] - lbbox[1]
        except AttributeError:
            lw, lh = draw.textsize(label, font=font_yolo)
        # 标签在框下方
        label_y = min(py2 + 2, H - lh - 4)
        draw.rectangle(
            [px1, label_y, px1 + lw + 6, label_y + lh + 4],
            fill=(220, 50, 50, 200),
        )
        draw.text((px1 + 3, label_y + 2), label, fill=(255, 255, 255, 255), font=font_yolo)

    # --- VLM 框 (绿色实线) ---
    vlm_color = (0, 200, 0)
    for obj in vlm_objects:
        bbox = obj.get("bbox", [])
        if len(bbox) != 4:
            continue
        # 归一化坐标 -> 图片像素
        px1 = bbox[0] * W
        py1 = bbox[1] * H
        px2 = bbox[2] * W
        py2 = bbox[3] * H

        draw.rectangle([px1, py1, px2, py2], outline=(0, 200, 0, 255), width=3)

        # 标签
        oid = obj.get("id", "?")
        cat = obj.get("category", "?")
        depth_zone = obj.get("depth_zone", "?")
        depth_m = obj.get("depth_m", "?")
        label = f"VLM#{oid} {cat} {depth_zone}{depth_m}m"
        try:
            lbbox = draw.textbbox((0, 0), label, font=font_vlm)
            lw = lbbox[2] - lbbox[0]
            lh = lbbox[3] - lbbox[1]
        except AttributeError:
            lw, lh = draw.textsize(label, font=font_vlm)
        # 标签在框上方
        label_y = max(py1 - lh - 4, 0)
        draw.rectangle(
            [px1, label_y, px1 + lw + 6, label_y + lh + 4],
            fill=(0, 160, 0, 200),
        )
        draw.text((px1 + 3, label_y + 2), label, fill=(255, 255, 255, 255), font=font_vlm)

    # 保存
    img.save(str(output_path), quality=95)
    print(f"  保存标注帧: {output_path.name} "
          f"(VLM={len(vlm_objects)}, YOLO={len(yolo_objects)})")
    return output_path


def make_grid_image(frame_paths: List[Path], output_path: Path, cols: int = 4):
    """
    将多张帧图缩放到 640x360 并拼成网格图。
    """
    from PIL import Image

    thumb_w, thumb_h = 640, 360
    n = len(frame_paths)
    if n == 0:
        print("  [警告] 无帧图可拼接网格")
        return
    rows = math.ceil(n / cols)

    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (30, 30, 30))
    for i, fp in enumerate(frame_paths):
        col = i % cols
        row = i // cols
        try:
            thumb = Image.open(str(fp)).convert("RGB")
            thumb = thumb.resize((thumb_w, thumb_h), Image.LANCZOS)
            grid.paste(thumb, (col * thumb_w, row * thumb_h))
        except Exception as e:
            print(f"  [警告] 网格帧 {fp.name} 加载失败: {e}")

    grid.save(str(output_path), quality=95)
    print(f"  保存网格图: {output_path.name} ({cols}x{rows}, {n} 帧)")


# ---------------------------------------------------------------------------
# IoU 对比分析
# ---------------------------------------------------------------------------

def match_vlm_yolo_per_frame(
    vlm_objects: List[dict],
    yolo_objects: List[dict],
    video_width: int,
    video_height: int,
) -> List[dict]:
    """
    对一帧内的 VLM 检测与 YOLO 检测做贪心 IoU 匹配。
    返回每个 VLM 目标的匹配结果列表。
    """
    results = []
    used_yolo = set()

    for vobj in vlm_objects:
        vbox = vobj.get("bbox", [])
        if len(vbox) != 4:
            results.append({
                "vlm_id": vobj.get("id", "?"),
                "vlm_category": vobj.get("category", "?"),
                "vlm_bbox": vbox,
                "best_yolo_id": None,
                "best_iou": 0.0,
                "depth_zone": vobj.get("depth_zone", "?"),
                "depth_m": vobj.get("depth_m", "?"),
            })
            continue

        best_iou = 0.0
        best_yolo_idx = None
        best_yolo_det = None

        for j, ydet in enumerate(yolo_objects):
            if j in used_yolo:
                continue
            ybbox_px = ydet.get("bbox_xyxy", [])
            if len(ybbox_px) != 4:
                continue
            ybbox_norm = normalize_yolo_bbox(ybbox_px, video_width, video_height)
            iou = compute_iou(vbox, ybbox_norm)
            if iou > best_iou:
                best_iou = iou
                best_yolo_idx = j
                best_yolo_det = ydet

        if best_yolo_idx is not None and best_iou > 0.05:
            used_yolo.add(best_yolo_idx)

        results.append({
            "vlm_id": vobj.get("id", "?"),
            "vlm_category": vobj.get("category", "?"),
            "vlm_bbox": vbox,
            "best_yolo_id": best_yolo_det.get("track_id") if best_yolo_det and best_iou > 0.05 else None,
            "best_yolo_category": best_yolo_det.get("category") if best_yolo_det and best_iou > 0.05 else None,
            "best_iou": round(best_iou, 4),
            "depth_zone": vobj.get("depth_zone", "?"),
            "depth_m": vobj.get("depth_m", "?"),
        })

    return results


# ---------------------------------------------------------------------------
# ID 稳定性分析
# ---------------------------------------------------------------------------

def analyze_vlm_track_stability(vlm_result: dict, total_frames: int) -> List[dict]:
    """分析 VLM 跟踪的 ID 连续性。"""
    track_summary = vlm_result.get("track_summary", {})
    results = []

    # 如果 track_summary 为空，从 frames 反推
    if not track_summary:
        id_frames: Dict[str, List[int]] = {}
        for frame_data in vlm_result.get("frames", []):
            fidx = frame_data.get("frame_idx", 0)
            for obj in frame_data.get("objects", []):
                oid = str(obj.get("id", "?"))
                id_frames.setdefault(oid, []).append(fidx)
        for oid, flist in sorted(id_frames.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
            flist_sorted = sorted(set(flist))
            gaps = sum(1 for i in range(1, len(flist_sorted)) if flist_sorted[i] - flist_sorted[i-1] > 1)
            results.append({
                "id": oid,
                "category": "?",
                "frames_present": flist_sorted,
                "num_frames": len(flist_sorted),
                "coverage": len(flist_sorted) / max(total_frames, 1),
                "continuous": gaps == 0,
                "gaps": gaps,
            })
        return results

    for oid, info in sorted(track_summary.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        frames_present = sorted(info.get("frames_present", []))
        num_frames = len(frames_present)
        gaps = sum(1 for i in range(1, num_frames) if frames_present[i] - frames_present[i-1] > 1)
        results.append({
            "id": oid,
            "category": info.get("category", "?"),
            "frames_present": frames_present,
            "num_frames": num_frames,
            "coverage": num_frames / max(total_frames, 1),
            "continuous": info.get("continuous", gaps == 0),
            "gaps": gaps,
        })
    return results


def analyze_yolo_track_stability(yolo_trajectories: dict, total_frames: int) -> List[dict]:
    """分析 YOLO/ByteTrack 跟踪的 ID 连续性。"""
    results = []
    for tid, info in sorted(yolo_trajectories.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999):
        frames = sorted(info.get("frame_indices", []))
        num_frames = len(frames)
        gaps = sum(1 for i in range(1, num_frames) if frames[i] - frames[i-1] > 1)
        results.append({
            "track_id": tid,
            "category": info.get("category", "?"),
            "frames_present": frames,
            "num_frames": num_frames,
            "coverage": num_frames / max(total_frames, 1),
            "gaps": gaps,
        })
    return results


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    input_data: dict,
    vlm_result: dict,
    iou_results_per_frame: Dict[int, List[dict]],
    vlm_tracks: List[dict],
    yolo_tracks: List[dict],
    output_path: Path,
):
    """生成 Markdown 对比报告。"""
    video_name = input_data.get("video_name", "unknown")
    escalation_reason = input_data.get("escalation_reason", "N/A")
    s1_verdict = input_data.get("s1_verdict", "N/A")
    s2_verdict = input_data.get("s2_verdict", "N/A")
    frames_data = vlm_result.get("frames", [])

    lines = []
    lines.append("# 3D Grounding vs YOLO 对比报告\n")

    # --- 视频信息 ---
    lines.append("## 视频信息\n")
    lines.append(f"- **视频**: {video_name}")
    lines.append(f"- **分辨率**: {input_data.get('video_width', '?')}x{input_data.get('video_height', '?')}")
    lines.append(f"- **S2 升级原因**: {escalation_reason}")
    lines.append(f"- **S1 判定**: {s1_verdict} -> S2 判定: {s2_verdict}")
    lines.append(f"- **帧数**: {len(frames_data)}\n")

    # --- VLM Grounding 结果摘要 ---
    lines.append("## VLM Grounding 结果摘要\n")
    lines.append("| 帧 | 时间戳 | 检测目标数 | ID列表 |")
    lines.append("|----|----|----|----|")
    for fd in frames_data:
        fidx = fd.get("frame_idx", "?")
        ts = fd.get("timestamp", 0)
        objs = fd.get("objects", [])
        ids = ",".join(str(o.get("id", "?")) for o in objs)
        lines.append(f"| {fidx} | {ts:.1f}s | {len(objs)} | {ids} |")
    lines.append("")

    # --- IoU 对比 ---
    lines.append("## YOLO vs VLM IoU 对比\n")
    lines.append("| 帧 | VLM ID | VLM类别 | 最佳匹配 YOLO ID | YOLO类别 | IoU | 深度估算 |")
    lines.append("|----|----|----|----|----|----|------|")
    all_ious = []
    matched_count = 0
    total_vlm_count = 0
    for fidx in sorted(iou_results_per_frame.keys()):
        for m in iou_results_per_frame[fidx]:
            total_vlm_count += 1
            yid = m.get("best_yolo_id")
            ycat = m.get("best_yolo_category", "-")
            iou_val = m["best_iou"]
            depth_str = f"{m.get('depth_zone', '?')}{m.get('depth_m', '?')}m"
            yid_str = str(yid) if yid is not None else "-"
            ycat_str = ycat if yid is not None else "-"
            lines.append(
                f"| {fidx} | {m['vlm_id']} | {m['vlm_category']} "
                f"| {yid_str} | {ycat_str} | {iou_val:.3f} | {depth_str} |"
            )
            if yid is not None:
                all_ious.append(iou_val)
                matched_count += 1
    lines.append("")

    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    lines.append(f"**匹配统计**: VLM 检测 {total_vlm_count} 个目标, "
                 f"匹配到 YOLO {matched_count} 个, 平均 IoU = {avg_iou:.3f}\n")

    # --- 跨帧 ID 稳定性 ---
    lines.append("## 跨帧 ID 稳定性对比\n")

    lines.append("### VLM Tracks\n")
    lines.append("| ID | 类别 | 出现帧数 | 覆盖率 | 是否连续 | 断裂次数 |")
    lines.append("|----|----|----|----|----|------|")
    vlm_continuous_count = 0
    for t in vlm_tracks:
        cont_str = "Yes" if t["continuous"] else "No"
        if t["continuous"]:
            vlm_continuous_count += 1
        lines.append(
            f"| {t['id']} | {t['category']} | {t['num_frames']} "
            f"| {t['coverage']:.1%} | {cont_str} | {t['gaps']} |"
        )
    vlm_cont_rate = vlm_continuous_count / max(len(vlm_tracks), 1)
    lines.append(f"\n**VLM track 连续率**: {vlm_cont_rate:.1%} "
                 f"({vlm_continuous_count}/{len(vlm_tracks)})\n")

    lines.append("### YOLO/ByteTrack Tracks\n")
    lines.append("| track_id | 类别 | 出现帧数 | 覆盖率 | 断裂次数 |")
    lines.append("|----|----|----|----|----|")
    for t in yolo_tracks:
        lines.append(
            f"| {t['track_id']} | {t['category']} | {t['num_frames']} "
            f"| {t['coverage']:.1%} | {t['gaps']} |"
        )
    lines.append("")

    # --- 深度估算分布 ---
    lines.append("## 深度估算分布\n")
    depth_dist: Dict[str, int] = {}
    for fd in frames_data:
        for obj in fd.get("objects", []):
            zone = obj.get("depth_zone", "未知")
            depth_dist[zone] = depth_dist.get(zone, 0) + 1
    lines.append("| 深度区域 | 检测数量 |")
    lines.append("|----|------|")
    for zone in ["近", "中", "远", "未知"]:
        if zone in depth_dist:
            lines.append(f"| {zone} | {depth_dist[zone]} |")
    lines.append("")

    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"  保存报告: {output_path.name}")
    return avg_iou, vlm_cont_rate, total_vlm_count, matched_count


def generate_timeline(
    vlm_result: dict,
    yolo_detections: dict,
    total_frames: int,
    output_path: Path,
):
    """生成时间线表格 Markdown。"""
    frame_indices = list(range(total_frames))

    # 收集 VLM tracks
    vlm_id_frames: Dict[str, set] = {}
    vlm_id_cat: Dict[str, str] = {}
    for fd in vlm_result.get("frames", []):
        fidx = fd.get("frame_idx", 0)
        for obj in fd.get("objects", []):
            oid = str(obj.get("id", "?"))
            vlm_id_frames.setdefault(oid, set()).add(fidx)
            vlm_id_cat[oid] = obj.get("category", "?")

    # 收集 YOLO tracks
    yolo_id_frames: Dict[str, set] = {}
    yolo_id_cat: Dict[str, str] = {}
    for fidx_str, dets in yolo_detections.items():
        fidx = int(fidx_str)
        for det in dets:
            tid = str(det.get("track_id", "?"))
            yolo_id_frames.setdefault(tid, set()).add(fidx)
            yolo_id_cat[tid] = det.get("category", "?")

    lines = []
    lines.append("# Grounding 时间线对比\n")

    # 表头
    header = "| 来源 | ID | 类别 |"
    sep = "|----|----|----|"
    for fi in frame_indices:
        header += f" F{fi} |"
        sep += "----|"
    lines.append(header)
    lines.append(sep)

    # VLM rows
    for oid in sorted(vlm_id_frames.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        row = f"| VLM | {oid} | {vlm_id_cat.get(oid, '?')} |"
        for fi in frame_indices:
            if fi in vlm_id_frames[oid]:
                row += f" V{oid} |"
            else:
                row += " |"
        lines.append(row)

    # YOLO rows
    for tid in sorted(yolo_id_frames.keys(), key=lambda x: int(x) if x.isdigit() else 999):
        row = f"| YOLO | {tid} | {yolo_id_cat.get(tid, '?')} |"
        for fi in frame_indices:
            if fi in yolo_id_frames[tid]:
                row += f" Y{tid} |"
            else:
                row += " |"
        lines.append(row)

    lines.append("")
    timeline_text = "\n".join(lines)
    output_path.write_text(timeline_text, encoding="utf-8")
    print(f"  保存时间线: {output_path.name}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VLM Grounding vs YOLO 对比测试")
    parser.add_argument(
        "--from-cache", action="store_true",
        help=f"跳过 VLM 调用，直接从 {OUTPUT_DIR}/grounding_vlm_raw.json 加载结果重新生成可视化"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  test_3d_grounding.py - VLM Grounding vs YOLO 对比测试")
    if args.from_cache:
        print("  [模式] 从缓存加载 VLM 结果（跳过 VLM 调用）")
    print("=" * 70)
    print()

    # 1. 检查输入文件
    if not INPUT_FILE.exists():
        print(f"[错误] 输入文件不存在: {INPUT_FILE}")
        print("  该文件由 Team A 生成，请先运行相应的预处理脚本。")
        sys.exit(1)

    print(f"[1/6] 读取输入文件: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    video_name = input_data.get("video_name", "unknown.mp4")
    video_width = input_data.get("video_width", 1920)
    video_height = input_data.get("video_height", 1080)
    s2_frames = input_data.get("s2_frames", [])
    yolo_detections = input_data.get("yolo_detections_per_frame", {})
    yolo_trajectories = input_data.get("yolo_trajectories", {})

    print(f"  视频: {video_name} ({video_width}x{video_height})")
    print(f"  S2 帧数: {len(s2_frames)}")
    print(f"  YOLO 检测帧数: {len(yolo_detections)}")
    print(f"  YOLO 轨迹数: {len(yolo_trajectories)}")
    print()

    # 检查帧图片存在性
    valid_frames = []
    for frame_info in s2_frames:
        img_path = resolve_frame_image_path(frame_info)
        if img_path is not None:
            valid_frames.append(frame_info)
        else:
            print(f"  [警告] 帧 {frame_info.get('frame_idx', '?')} 图片不存在，跳过")
    if not valid_frames:
        print("[错误] 所有帧图片都不存在，无法继续")
        sys.exit(1)
    print(f"  有效帧数: {len(valid_frames)}/{len(s2_frames)}")
    print()

    # 2. 调用 VLM（或从缓存加载）
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    vlm_raw_path = OUTPUT_DIR / "grounding_vlm_raw.json"

    if args.from_cache:
        print(f"[2/6] 从缓存加载 VLM 结果: {vlm_raw_path}")
        if not vlm_raw_path.exists():
            print(f"[错误] 缓存文件不存在: {vlm_raw_path}")
            print("  请先不带 --from-cache 参数运行一次完整流程")
            sys.exit(1)
        with open(vlm_raw_path, "r", encoding="utf-8") as f:
            vlm_result = json.load(f)
        print(f"  已加载: {len(vlm_result.get('frames', []))} 帧, "
              f"{len(vlm_result.get('track_summary', {}))} 个 track")
    else:
        print(f"[2/6] 调用 VLM Grounding ({VLLM_MODEL} @ {VLLM_BASE_URL})")
        vlm_result = call_vlm_grounding(valid_frames)
        if vlm_result is None:
            print("[错误] VLM 调用失败，生成空结果继续分析")
            vlm_result = {"frames": [], "track_summary": {}}
        print()

        # 保存 VLM 原始结果
        with open(vlm_raw_path, "w", encoding="utf-8") as f:
            json.dump(vlm_result, f, ensure_ascii=False, indent=2)
        print(f"  VLM 原始结果已保存: {vlm_raw_path.name}")

    # --- 自适应坐标归一化 VLM bbox ---
    # Qwen-VL 内部会将图片 resize 到自身的 patch grid（不保证等于我们发送的尺寸）。
    # 实验发现：发送 640x337 → VLM 内部坐标空间约 914x709。
    # 用自适应方案：先扫描所有输出 bbox 找最大 x/y，再加 5% 余量作为归一化基准。
    _all_vlm_x: List[float] = []
    _all_vlm_y: List[float] = []
    for _fd in vlm_result.get("frames", []):
        for _obj in _fd.get("objects", []):
            _bb = _obj.get("bbox", [])
            if len(_bb) == 4:
                _all_vlm_x.extend([_bb[0], _bb[2]])
                _all_vlm_y.extend([_bb[1], _bb[3]])

    _pixel_bbox_count = 0
    if _all_vlm_x and max(_all_vlm_x) > 1.0:
        # 像素坐标 → 自适应归一化
        _vlm_space_w = max(_all_vlm_x) * 1.05   # 5% 余量防止 bbox 恰好触边
        _vlm_space_h = max(_all_vlm_y) * 1.05
        print(f"  VLM 自适应坐标空间检测: {_vlm_space_w:.0f}x{_vlm_space_h:.0f} px")
        print(f"  (发送给 VLM 的图片约 640x{int(video_height * 640 / video_width)} px，"
              f"模型内部 grid 自动放大)")
        for _fd in vlm_result.get("frames", []):
            for _obj in _fd.get("objects", []):
                _bb = _obj.get("bbox", [])
                if len(_bb) == 4 and any(v > 1.0 for v in _bb):
                    _obj["bbox"] = [
                        max(0.0, min(1.0, _bb[0] / _vlm_space_w)),
                        max(0.0, min(1.0, _bb[1] / _vlm_space_h)),
                        max(0.0, min(1.0, _bb[2] / _vlm_space_w)),
                        max(0.0, min(1.0, _bb[3] / _vlm_space_h)),
                    ]
                    _pixel_bbox_count += 1
        if _pixel_bbox_count:
            print(f"  自适应归一化完成: {_pixel_bbox_count} 个 bbox")
    else:
        print("  VLM bbox 已是归一化坐标（0-1），无需转换")
    print()

    # 3. 绘制标注图
    print(f"[3/6] 绘制标注帧图片")
    total_frames = len(valid_frames)
    vlm_frames_data = vlm_result.get("frames", [])

    # 建立 VLM frame_idx -> objects 的映射
    vlm_objects_map: Dict[int, List[dict]] = {}
    for fd in vlm_frames_data:
        fidx = fd.get("frame_idx", -1)
        vlm_objects_map[fidx] = fd.get("objects", [])

    annotated_paths: List[Path] = []
    for i, frame_info in enumerate(valid_frames):
        fidx = frame_info.get("frame_idx", i)
        ts = frame_info.get("timestamp", 0.0)
        img_path = resolve_frame_image_path(frame_info)

        vlm_objs = vlm_objects_map.get(fidx, [])
        # 也尝试按序号匹配（VLM 可能返回不同的 frame_idx）
        if not vlm_objs and i < len(vlm_frames_data):
            vlm_objs = vlm_frames_data[i].get("objects", [])

        yolo_objs = yolo_detections.get(str(fidx), [])

        out_path = OUTPUT_DIR / f"grounding_frame_{fidx:02d}.jpg"
        draw_frame_annotations(
            img_path=img_path,
            frame_idx=fidx,
            timestamp=ts,
            vlm_objects=vlm_objs,
            yolo_objects=yolo_objs,
            video_width=video_width,
            video_height=video_height,
            output_path=out_path,
        )
        annotated_paths.append(out_path)

    print()

    # 4. 生成网格图
    print(f"[4/6] 生成网格总览图")
    grid_path = OUTPUT_DIR / "grounding_grid.jpg"
    make_grid_image(annotated_paths, grid_path, cols=4)
    print()

    # 5. IoU 对比 + ID 稳定性分析
    print(f"[5/6] IoU 对比 + ID 稳定性分析")
    iou_results_per_frame: Dict[int, List[dict]] = {}
    for i, frame_info in enumerate(valid_frames):
        fidx = frame_info.get("frame_idx", i)
        vlm_objs = vlm_objects_map.get(fidx, [])
        if not vlm_objs and i < len(vlm_frames_data):
            vlm_objs = vlm_frames_data[i].get("objects", [])
        yolo_objs = yolo_detections.get(str(fidx), [])
        matches = match_vlm_yolo_per_frame(vlm_objs, yolo_objs, video_width, video_height)
        iou_results_per_frame[fidx] = matches

    vlm_tracks = analyze_vlm_track_stability(vlm_result, total_frames)

    # 从 yolo_detections_per_frame（帧级检测）重建 frame_indices，
    # 因为 yolo_trajectories 里用 frame_timestamps 而非 frame_indices
    yolo_traj_by_frame: Dict[str, Dict] = {}
    for _fidx_str, _dets in yolo_detections.items():
        _fidx = int(_fidx_str)
        for _det in _dets:
            _tid = str(_det.get("track_id", "?"))
            if _tid not in yolo_traj_by_frame:
                yolo_traj_by_frame[_tid] = {
                    "category": _det.get("category", "?"),
                    "frame_indices": [],
                }
            if _fidx not in yolo_traj_by_frame[_tid]["frame_indices"]:
                yolo_traj_by_frame[_tid]["frame_indices"].append(_fidx)

    yolo_tracks = analyze_yolo_track_stability(yolo_traj_by_frame, total_frames)
    print(f"  VLM tracks: {len(vlm_tracks)}, YOLO tracks: {len(yolo_tracks)}")
    print()

    # 6. 生成报告
    print(f"[6/6] 生成对比报告")
    report_path = OUTPUT_DIR / "grounding_vs_yolo_report.md"
    avg_iou, vlm_cont_rate, total_vlm, matched = generate_report(
        input_data, vlm_result, iou_results_per_frame,
        vlm_tracks, yolo_tracks, report_path,
    )

    timeline_path = OUTPUT_DIR / "grounding_track_timeline.md"
    generate_timeline(vlm_result, yolo_detections, total_frames, timeline_path)
    print()

    # --- 汇总 ---
    print("=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  总帧数:              {total_frames}")
    print(f"  VLM 检测目标总数:    {total_vlm}")
    print(f"  匹配到 YOLO 的数量:  {matched}")
    print(f"  平均 IoU:            {avg_iou:.3f}")
    print(f"  VLM track 数:        {len(vlm_tracks)}")
    print(f"  VLM track 连续率:    {vlm_cont_rate:.1%}")
    print(f"  YOLO track 数:       {len(yolo_tracks)}")
    yolo_gap_total = sum(t["gaps"] for t in yolo_tracks)
    print(f"  YOLO track 总断裂数: {yolo_gap_total}")
    print()
    print("  输出文件:")
    print(f"    - 标注帧:   outputs/grounding_frame_*.jpg ({len(annotated_paths)} 张)")
    print(f"    - 网格图:   {grid_path.name}")
    print(f"    - 对比报告: {report_path.name}")
    print(f"    - 时间线:   {timeline_path.name}")
    print(f"    - VLM 原始: {vlm_raw_path.name}")
    print()
    print("完成。")


if __name__ == "__main__":
    main()
