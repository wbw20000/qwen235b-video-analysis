from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


def _cut_clip_task(args: Tuple[str, float, float, str, str, dict]) -> dict:
    """
    用于多进程执行的剪辑任务（需要在模块级别定义以支持pickle）

    Args:
        args: (input_path, start, end, output_path, clip_id, clip_dict)

    Returns:
        包含剪辑结果的字典
    """
    input_path, start, end, output_path, clip_id, clip_extras = args

    result = {
        "clip_id": clip_id,
        "success": False,
        "video_path": None,
        "clip_source": None,
        "error": None,
        "extras": clip_extras  # 保留额外的clip信息（如is_accident, long_version等）
    }

    try:
        success = cut_clip_ffmpeg(input_path, start, end, output_path)
        if success:
            result["success"] = True
            result["video_path"] = output_path
            result["clip_source"] = "clipped"
        else:
            result["video_path"] = input_path
            result["clip_source"] = "original_fallback"
            result["error"] = "剪辑失败"
    except Exception as e:
        result["video_path"] = input_path
        result["clip_source"] = "original_fallback"
        result["error"] = str(e)

    return result


def parallel_cut_clips(
    clips: List[dict],
    video_path: str,
    output_dir: str,
    max_workers: int = 4
) -> Tuple[List[dict], int, int]:
    """
    并行剪辑多个视频片段

    Args:
        clips: clip信息列表，每个必须包含 clip_id, start_time, end_time
        video_path: 源视频路径
        output_dir: 输出目录
        max_workers: 最大并行进程数

    Returns:
        (更新后的clips列表, 成功数, 失败数)
    """
    print(f"[clip_sampler] 启动{max_workers}进程并行剪辑，共{len(clips)}个片段")

    # 准备任务参数
    tasks = []
    for clip in clips:
        out_path = os.path.join(output_dir, f"{clip['clip_id']}.mp4")
        extras = {
            "is_accident": clip.get("is_accident", False),
            "long_version": clip.get("long_version"),
            "accident_score": clip.get("accident_score", 0),
        }
        tasks.append((
            video_path,
            clip["start_time"],
            clip["end_time"],
            out_path,
            clip["clip_id"],
            extras
        ))

    # 并行执行
    results = {}
    success_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_cut_clip_task, task): task[4] for task in tasks}  # task[4] = clip_id

        for future in as_completed(futures):
            clip_id = futures[future]
            try:
                result = future.result()
                results[clip_id] = result
                if result["success"]:
                    success_count += 1
                    print(f"[clip_sampler] ✓ {clip_id} 剪辑完成")
                else:
                    fail_count += 1
                    print(f"[clip_sampler] ✗ {clip_id} 剪辑失败: {result.get('error', '未知')}")
            except Exception as e:
                fail_count += 1
                results[clip_id] = {
                    "clip_id": clip_id,
                    "success": False,
                    "video_path": video_path,
                    "clip_source": "original_fallback",
                    "error": str(e)
                }
                print(f"[clip_sampler] ✗ {clip_id} 进程异常: {e}")

    # 更新原始clips列表
    updated_clips = []
    for clip in clips:
        clip_id = clip["clip_id"]
        if clip_id in results:
            result = results[clip_id]
            clip["video_path"] = result["video_path"]
            clip["clip_source"] = result["clip_source"]
        else:
            clip["video_path"] = video_path
            clip["clip_source"] = "original_fallback"
        updated_clips.append(clip)

    print(f"[clip_sampler] 并行剪辑完成: 成功{success_count}, 失败{fail_count}")
    return updated_clips, success_count, fail_count


def cut_clip_ffmpeg(input_path: str, start: float, end: float, output_path: str) -> bool:
    """使用 ffmpeg 裁剪片段，保留编码。"""
    duration = max(0.1, end - start)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts", "make_zero",  # 避免时间戳问题
        output_path,
    ]

    print(f"[clip_sampler] 剪辑视频: {start:.2f}s - {end:.2f}s (时长: {duration:.2f}s)")
    print(f"[clip_sampler] 输出路径: {output_path}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=os.name == "nt",
            timeout=60  # 添加超时
        )

        # 验证输出文件是否存在且有效
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            output_size = os.path.getsize(output_path) / 1024  # KB
            print(f"[clip_sampler] ✓ 剪辑成功，文件大小: {output_size:.1f} KB")
            return True
        else:
            print(f"[clip_sampler] ✗ 剪辑失败：输出文件不存在或为空")
            return False

    except subprocess.TimeoutExpired:
        print(f"[clip_sampler] ✗ 剪辑超时")
        return False
    except subprocess.CalledProcessError as e:
        print(f"[clip_sampler] ✗ FFmpeg 错误: {e.stderr.decode('utf-8', errors='ignore') if e.stderr else '未知错误'}")
        return False
    except Exception as e:
        print(f"[clip_sampler] ✗ 剪辑异常: {str(e)}")
        return False


def sample_frames_from_clip(
    video_path: str, start: float, end: float, num_frames: int,
    min_frames_required: int = 6,
    max_retry: int = 2,
) -> List[Tuple[float, "cv2.Mat"]]:
    """
    使用FFmpeg批量抽帧（比逐帧seek更高效）。
    返回 [(timestamp, frame_bgr)]。

    [B修复] 添加ensure_min_frames兜底：
    - 如果解码帧数不足，自动提高fps重试
    - 最多重试max_retry次

    Args:
        video_path: 视频路径
        start: 开始时间（秒）
        end: 结束时间（秒）
        num_frames: 请求帧数
        min_frames_required: 最低帧数要求（默认6）
        max_retry: 最大重试次数

    Returns:
        [(timestamp, frame_bgr), ...]
    """
    if num_frames <= 0:
        return []

    duration = max(0.1, end - start)

    # 计算采样时间点
    if num_frames == 1:
        positions = [start + duration / 2]  # 单帧取中间
    else:
        positions = [start + i * duration / (num_frames - 1) for i in range(num_frames)]

    retry_count = 0
    decoded_n = 0
    final_fps = 0.0

    # [B修复] 重试循环
    while retry_count <= max_retry:
        # 使用临时目录存储抽取的帧
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_pattern = os.path.join(tmp_dir, "frame_%04d.jpg")

            # 计算目标fps：确保在duration内能抽取num_frames帧
            # 重试时提高fps
            base_fps = num_frames / duration if duration > 0 else num_frames
            fps_multiplier = 1.0 + retry_count * 0.5  # 每次重试提高50%
            min_fps = 2.0 * fps_multiplier
            final_fps = max(min_fps, base_fps * fps_multiplier)

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-vf", f"fps={final_fps:.6f}",
                "-frames:v", str(int(num_frames * fps_multiplier)),  # 请求更多帧
                "-q:v", "2",  # 高质量JPEG
                output_pattern,
            ]

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=os.name == "nt",
                    timeout=60
                )
            except subprocess.TimeoutExpired:
                print(f"[clip_sampler] FFmpeg抽帧超时，回退到OpenCV")
                samples = _sample_frames_opencv_fallback(video_path, start, end, num_frames)
                print(f"[clip_sampler] 抽帧统计: requested={num_frames}, decoded={len(samples)}, retry={retry_count}, fps={final_fps:.2f}")
                return samples
            except subprocess.CalledProcessError as e:
                stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
                print(f"[clip_sampler] FFmpeg抽帧失败: {stderr[:200]}，回退到OpenCV")
                samples = _sample_frames_opencv_fallback(video_path, start, end, num_frames)
                print(f"[clip_sampler] 抽帧统计: requested={num_frames}, decoded={len(samples)}, retry={retry_count}, fps={final_fps:.2f}")
                return samples
            except Exception as e:
                print(f"[clip_sampler] FFmpeg抽帧异常: {e}，回退到OpenCV")
                samples = _sample_frames_opencv_fallback(video_path, start, end, num_frames)
                print(f"[clip_sampler] 抽帧统计: requested={num_frames}, decoded={len(samples)}, retry={retry_count}, fps={final_fps:.2f}")
                return samples

            # 读取提取的帧
            samples = []
            frame_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.jpg')])
            for i, fname in enumerate(frame_files[:num_frames]):  # 最多取num_frames帧
                frame_path = os.path.join(tmp_dir, fname)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    ts = positions[min(i, len(positions) - 1)]
                    samples.append((ts, frame))

            decoded_n = len(samples)

            # [B修复] 检查是否满足最低帧数要求
            if decoded_n >= min(min_frames_required, num_frames):
                print(f"[clip_sampler] 抽帧成功: requested={num_frames}, decoded={decoded_n}, retry={retry_count}, fps={final_fps:.2f}")
                return samples

            # 帧数不足，准备重试
            retry_count += 1
            if retry_count <= max_retry:
                print(f"[clip_sampler] ⚠ 帧数不足({decoded_n}<{min_frames_required})，重试#{retry_count} (提高fps)")

    # 所有重试都失败，回退到OpenCV
    print(f"[clip_sampler] 重试{max_retry}次仍不足，回退到OpenCV")
    samples = _sample_frames_opencv_fallback(video_path, start, end, num_frames)
    print(f"[clip_sampler] 抽帧统计: requested={num_frames}, decoded={len(samples)}, retry={retry_count}, fps={final_fps:.2f}")
    return samples


def _sample_frames_opencv_fallback(
    video_path: str, start: float, end: float, num_frames: int
) -> List[Tuple[float, "cv2.Mat"]]:
    """
    OpenCV回退方法（FFmpeg失败时使用）。
    逐帧seek，效率较低但兼容性好。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = max(0.1, end - start)
    positions = [start + i * duration / max(1, num_frames - 1) for i in range(num_frames)]

    samples = []
    for ts in positions:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            samples.append((ts, frame))
    cap.release()

    print(f"[clip_sampler] OpenCV回退抽帧完成: {len(samples)}帧")
    return samples
