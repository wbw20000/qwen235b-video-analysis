"""
Phase B - Step 1: 从已确认 TP 视频中提取 exemplar 帧
输出到 data/accident_exemplars_candidates/{roadside,elec_police,general}/
并生成 preview.html 供人工确认

用法:
    python tools/extract_accident_exemplars.py
    python tools/extract_accident_exemplars.py --max-per-category 8
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CACHE_DIR = BASE_DIR / "data" / "video_results"
OUT_DIR = BASE_DIR / "data" / "accident_exemplars_candidates"
VIDEO_DIR = BASE_DIR / "uploads" / "大样本事故数据集"
CAR_NET_DIR = VIDEO_DIR / "车网路测事故"

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"

# ── 摄像机类型判断 ──────────────────────────────────────────
def detect_camera_type(video_name: str) -> str:
    name = video_name.lower()
    if "roadsidecamera" in name or name.startswith("rc_") or name.startswith("rc.") or "_rc." in name:
        return "roadside"
    if re.match(r"^\d{2,3}-\d{3}", name):
        return "elec_police"
    return "general"


# ── 从结果缓存中提取 keyframe 路径 ────────────────────────────
def load_tp_keyframes_from_cache(max_per_category: int):
    """从 YES/POST_EVENT_ONLY 视频缓存中提取 keyframe 路径，按摄像机分类"""
    category_frames = {"roadside": [], "elec_police": [], "general": []}

    gz_files = sorted(CACHE_DIR.glob("*.result.json.gz"))
    print(f"[extract] 扫描 {len(gz_files)} 个缓存文件...")

    for gz in gz_files:
        try:
            with gzip.open(gz, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        # 取 perf_stats 里的 video_path 判断摄像机类型
        video_path = d.get("perf_stats", {}).get("video_path", "") or str(gz)
        video_name = os.path.basename(video_path)
        cam_type = detect_camera_type(video_name)

        if len(category_frames[cam_type]) >= max_per_category:
            continue

        # 检查是否有 YES 或 POST_EVENT 的 result
        results = d.get("results", [])
        has_positive = False
        for r in results:
            vout = r.get("vlm_output", {}) or {}
            if vout.get("verdict") in ("YES", "POST_EVENT_ONLY"):
                has_positive = True
                break

        # 也检查 narrative_sidecar
        sidecar = d.get("narrative_sidecar", {})
        if sidecar.get("triggered"):
            for m_data in sidecar.get("models", {}).values():
                if m_data.get("accident_status") in ("CONFIRMED", "SUSPECTED"):
                    has_positive = True
                    break

        if not has_positive:
            continue

        # 收集 keyframe 路径（最多取 3 帧）
        kframes = d.get("keyframes", [])
        paths = [BASE_DIR / k["path"] for k in kframes if k.get("path")]
        valid = [p for p in paths if p.exists()][:3]

        if valid:
            category_frames[cam_type].append({
                "video": video_name,
                "paths": valid,
                "cam_type": cam_type,
            })

    return category_frames


# ── 从视频直接用 ffmpeg 提取帧 ──────────────────────────────────
def extract_frames_from_video(video_path: Path, out_dir: Path, n_frames: int = 4) -> list[Path]:
    """用 ffmpeg 均匀提取 n_frames 帧"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 获取视频时长（捕获 bytes，手动 decode 避免 Windows 编码问题）
    try:
        result = subprocess.run(
            [FFMPEG, "-i", str(video_path)],
            capture_output=True  # bytes 模式
        )
        stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr_text)
        if not duration_match:
            print(f"  [ffmpeg] 无法解析时长: {video_path.name[:40]}")
            return []
        h = int(duration_match.group(1))
        m = int(duration_match.group(2))
        s = float(duration_match.group(3))
        duration = h * 3600 + m * 60 + s
    except Exception as e:
        print(f"  [ffmpeg] 获取时长失败: {e}")
        return []

    if duration < 1.0:
        print(f"  [ffmpeg] 视频时长过短: {duration:.1f}s")
        return []

    # 均匀采样时间点（避开首尾 10%）
    margin = min(duration * 0.1, 10.0)
    usable = duration - 2 * margin
    step = usable / (n_frames - 1) if n_frames > 1 else 0
    timestamps = [margin + i * step for i in range(n_frames)]

    extracted = []
    stem = video_path.stem[:30]
    for i, ts in enumerate(timestamps):
        out_path = out_dir / f"{stem}_t{int(ts):04d}_f{i+1}.jpg"
        try:
            subprocess.run(
                [FFMPEG, "-ss", f"{ts:.2f}", "-i", str(video_path),
                 "-vframes", "1", "-q:v", "3", str(out_path), "-y"],
                capture_output=True, check=True
            )
            if out_path.exists() and out_path.stat().st_size > 1000:
                extracted.append(out_path)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="replace")[-200:]
            print(f"  [ffmpeg] 提取帧失败 t={ts:.1f}: {err}")
        except Exception as e:
            print(f"  [ffmpeg] 异常: {e}")

    return extracted


# ── 生成 HTML 预览 ─────────────────────────────────────────────
def generate_preview_html(out_dir: Path, candidates: dict):
    """生成供人工审阅的 HTML"""
    html_parts = ["""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Exemplar Candidates Preview</title>
<style>
body { font-family: Arial, sans-serif; background: #1a1a1a; color: #eee; margin: 20px; }
h1 { color: #ffd700; }
h2 { color: #90caf9; border-bottom: 1px solid #444; padding-bottom: 6px; }
.video-block { margin: 20px 0; background: #2a2a2a; padding: 12px; border-radius: 8px; }
.video-title { font-size: 13px; color: #aaa; margin-bottom: 8px; }
.frames { display: flex; flex-wrap: wrap; gap: 8px; }
.frame-card { text-align: center; }
.frame-card img { max-width: 280px; max-height: 200px; border: 2px solid #555;
                   border-radius: 4px; cursor: pointer; }
.frame-card img:hover { border-color: #ffd700; }
.frame-label { font-size: 11px; color: #888; margin-top: 4px; }
.instructions { background: #333; padding: 12px; border-radius: 8px; margin-bottom: 20px; color: #cfc; }
</style>
</head>
<body>
<h1>Accident Exemplar Candidates</h1>
<div class="instructions">
  <b>操作说明：</b><br>
  1. 浏览每个分类中的候选帧，识别最能代表该类事故的帧<br>
  2. 将选中的帧从 <code>data/accident_exemplars_candidates/</code> 复制到
     <code>data/accident_exemplars/{roadside|elec_police|general}/</code><br>
  3. 每类建议保留 5-8 张最典型的帧（展示不同严重程度和视角）<br>
  4. 确认后告知 Claude 继续执行 Phase B-Step3
</div>
"""]

    cam_labels = {
        "roadside": "RoadsideCamera（俯视/远景，目标小）",
        "elec_police": "电警摄像机（平视/中距离）",
        "general": "通用摄像机",
    }

    for cam_type, items in candidates.items():
        if not items:
            continue
        label = cam_labels.get(cam_type, cam_type)
        html_parts.append(f"<h2>{label}（{len(items)} 个视频，共 {sum(len(x['paths']) for x in items)} 帧）</h2>")

        for item in items:
            video_short = item["video"][:60]
            html_parts.append(f'<div class="video-block">')
            html_parts.append(f'<div class="video-title">{video_short}</div>')
            html_parts.append('<div class="frames">')
            for img_path in item["paths"]:
                rel = os.path.relpath(img_path, out_dir)
                fname = os.path.basename(img_path)
                html_parts.append(
                    f'<div class="frame-card">'
                    f'<img src="{rel}" title="{fname}" onclick="this.style.borderColor=\'#0f0\'">'
                    f'<div class="frame-label">{fname[:40]}</div>'
                    f'</div>'
                )
            html_parts.append('</div></div>')

    html_parts.append("</body></html>")

    html_path = out_dir / "preview.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    return html_path


# ── 主流程 ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-category", type=int, default=10,
                        help="每类最多从缓存取多少个视频的帧")
    parser.add_argument("--include-carnet-videos", action="store_true", default=True,
                        help="从车网TP视频直接提取帧（用ffmpeg）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("roadside", "elec_police", "general"):
        (OUT_DIR / sub).mkdir(exist_ok=True)

    print("=" * 60)
    print("Phase B Step 1: 提取 Exemplar 候选帧")
    print("=" * 60)

    # ── 1. 从缓存里拿 keyframes ──────────────────────────────
    print("\n[1/3] 从现有缓存结果提取 keyframe 路径...")
    category_frames = load_tp_keyframes_from_cache(args.max_per_category)
    for cat, items in category_frames.items():
        print(f"  {cat}: {len(items)} 个视频")

    # ── 2. 车网 TP 视频 → ffmpeg 直接提帧 ───────────────────
    print("\n[2/3] 从车网 TP 视频直接提取帧（ffmpeg）...")
    # 确认 TP 的车网视频（非 FN）
    car_net_tp = [
        "25-107荣京西街与天宝中路南向北电警_20251231171435680.mp4",
        "RoadsideCamera.108_N-BD02VO_20260203090100_20260203090600.mp4",
        "RoadsideCamera.112_N-BD017V_20260204142100_20260204142600.mp4",
        "RoadsideCamera.113_N-BD0184_20260206145550_20260206150050.mp4",
        "RoadsideCamera.164_N-BD0374_20260207112330_20260207112830.mp4",
        "RoadsideCamera.255_N-BD027P_20260114164800_20260114165300.mp4",
        "RoadsideCamera.35_N-BD00FB_20260206134900_20260206135400.mp4",
        "RoadsideCamera.390_N-MN054E_20260206182500_20260206183000.mp4",
        "RoadsideCamera.396_N-MN00B2_20260202091000_20260202091500.mp4",
    ]
    for vname in car_net_tp:
        vpath = CAR_NET_DIR / vname
        if not vpath.exists():
            print(f"  [SKIP] 文件不存在: {vname[:50]}")
            continue
        cam_type = detect_camera_type(vname)
        sub_dir = OUT_DIR / cam_type
        frames = extract_frames_from_video(vpath, sub_dir, n_frames=4)
        print(f"  [{cam_type}] {vname[:50]} → {len(frames)} 帧")
        if frames:
            # 追加到 category_frames 用于 HTML 预览
            category_frames[cam_type].append({
                "video": vname,
                "paths": frames,
                "cam_type": cam_type,
            })

    # ── 3. 把缓存 keyframe 复制到候选目录 ───────────────────
    print("\n[3/3] 复制缓存 keyframe 到候选目录...")
    for cam_type, items in category_frames.items():
        for item in items:
            if not item["paths"]:
                continue
            # 仅复制不是已经在 out_dir 内的帧（ffmpeg 已直接输出到子目录）
            for p in item["paths"]:
                p = Path(p)
                dest_dir = OUT_DIR / cam_type
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / p.name
                if dest != p and not dest.exists():
                    try:
                        shutil.copy2(p, dest)
                        item["paths"] = [dest if pp == p else pp for pp in item["paths"]]
                    except Exception as e:
                        print(f"    复制失败: {p.name}: {e}")

    # ── 生成 HTML 预览 ────────────────────────────────────
    print("\n[生成预览 HTML]...")
    html_path = generate_preview_html(OUT_DIR, category_frames)
    print(f"\n{'='*60}")
    print(f"完成！")
    for cat, items in category_frames.items():
        total_frames = sum(len(x["paths"]) for x in items)
        print(f"  {cat}: {len(items)} 个视频, {total_frames} 帧")
    print(f"\n预览 HTML: {html_path}")
    print(f"候选帧目录: {OUT_DIR}")
    print(f"\n请打开 preview.html 在浏览器中审阅，将选中帧复制到 data/accident_exemplars/")
    print(f"完成后通知 Claude 继续 Phase B-Step3")


if __name__ == "__main__":
    main()
