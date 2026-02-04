#!/usr/bin/env python3
"""
事故视频剪辑器 - 自动剪辑事故片段并生成报告

监控分析结果，检测到事故时：
1. 剪辑事故视频片段（事故时间前后各30秒）
2. 生成事故分析报告（Markdown格式）
3. 保存关键帧截图
4. 统一存储到 /data1/accidents/ 目录
"""
import os
import sys
import json
import gzip
import time
import subprocess
from datetime import datetime
from pathlib import Path

# 配置
RESULTS_DIR = "/data1/results"
ACCIDENTS_DIR = "/data1/accidents"
RECORDINGS_DIR = "/data1/videos/rtsp_recordings"
WINDOWS_DIR = "/data1/videos/windows"

# 事故片段参数
PRE_ACCIDENT_SECONDS = 30   # 事故前保留秒数
POST_ACCIDENT_SECONDS = 30  # 事故后保留秒数

# 摄像头信息映射
CAMERA_INFO = {
    "147-A": {
        "location": "科创十七街与通惠干渠路",
        "direction": "北向南",
        "device_type": "电警"
    },
    "147-G": {
        "location": "科创十七街与通惠干渠路",
        "direction": "北向南",
        "device_type": "卡口"
    },
    "146-A": {
        "location": "科创十七街与经海路",
        "direction": "北向南",
        "device_type": "电警"
    }
}


def get_camera_info(camera_id: str) -> dict:
    """获取摄像头信息"""
    return CAMERA_INFO.get(camera_id, {
        "location": "未知路口",
        "direction": "未知方向",
        "device_type": "未知设备"
    })


def clip_accident_video(source_video: str, output_path: str,
                        start_time: float = 0, duration: float = 60) -> bool:
    """剪辑事故视频片段"""
    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", source_video,
            "-t", str(duration),
            "-c", "copy",  # 无损剪辑
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"剪辑失败: {e}")
        return False


def extract_keyframes(video_path: str, output_dir: str, num_frames: int = 5) -> list:
    """提取关键帧截图"""
    frames = []
    try:
        # 获取视频时长
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())

        # 均匀提取帧
        interval = duration / (num_frames + 1)
        for i in range(1, num_frames + 1):
            timestamp = interval * i
            frame_path = os.path.join(output_dir, f"keyframe_{i:02d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(timestamp),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                frame_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(frame_path):
                frames.append(frame_path)
    except Exception as e:
        print(f"提取关键帧失败: {e}")
    return frames


def generate_accident_report(result_data: dict, camera_info: dict,
                             accident_dir: str, video_filename: str,
                             keyframes: list) -> str:
    """生成事故分析报告（Markdown格式）"""

    # 解析时间
    event_time = result_data.get("event_time", "")
    if event_time:
        try:
            dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y年%m月%d日 %H:%M:%S")
        except:
            time_str = event_time
    else:
        time_str = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    camera_id = result_data.get("camera_id", "未知")
    vlm_verdict = result_data.get("vlm_verdict", "")
    confidence = result_data.get("confidence", 0)
    vlm_response = result_data.get("vlm_response", "")

    # Sidecar 分析结果
    sidecar = result_data.get("sidecar", {})
    narrative = sidecar.get("narrative", "")
    severity = sidecar.get("severity", "未评估")
    involved_objects = sidecar.get("involved_objects", [])

    report = f"""# 交通事故分析报告

## 基本信息

| 项目 | 内容 |
|------|------|
| **事故时间** | {time_str} |
| **事故地点** | {camera_info['location']} |
| **行驶方向** | {camera_info['direction']} |
| **监控设备** | {camera_info['device_type']} ({camera_id}) |
| **检测置信度** | {confidence:.1%} |
| **严重程度** | {severity} |

---

## 事故分析

### VLM 判定结果

**判定**: {vlm_verdict}

**分析说明**:
{vlm_response}

### 事故描述

{narrative if narrative else "暂无详细描述"}

### 涉及对象

"""

    if involved_objects:
        for obj in involved_objects:
            report += f"- {obj}\n"
    else:
        report += "- 暂无详细信息\n"

    report += f"""
---

## 证据材料

### 事故视频

- 文件名: `{video_filename}`
- 存储路径: `{accident_dir}/`

### 关键帧截图

"""

    for i, frame in enumerate(keyframes, 1):
        frame_name = os.path.basename(frame)
        report += f"![关键帧{i}]({frame_name})\n\n"

    report += f"""
---

## 原始数据

### 分析结果 JSON

详见 `result.json` 文件

---

*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*由 Traffic VLM 系统自动生成*
"""

    return report


def process_accident(result_file: str) -> bool:
    """处理单个事故结果"""
    try:
        # 读取结果文件
        with gzip.open(result_file, "rt", encoding="utf-8") as f:
            result_data = json.load(f)

        # 检查是否为事故
        if not result_data.get("is_accident", False):
            return False

        camera_id = result_data.get("camera_id", "unknown")
        event_time = result_data.get("event_time", "")
        video_path = result_data.get("video_path", "")

        # 生成事故ID和目录
        if event_time:
            try:
                dt = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y%m%d")
                time_str = dt.strftime("%H%M%S")
            except:
                date_str = datetime.now().strftime("%Y%m%d")
                time_str = datetime.now().strftime("%H%M%S")
        else:
            date_str = datetime.now().strftime("%Y%m%d")
            time_str = datetime.now().strftime("%H%M%S")

        accident_id = f"{camera_id}_{date_str}_{time_str}"
        accident_dir = os.path.join(ACCIDENTS_DIR, camera_id, date_str, accident_id)

        # 检查是否已处理
        if os.path.exists(accident_dir):
            print(f"事故已处理: {accident_id}")
            return False

        # 创建事故目录
        os.makedirs(accident_dir, exist_ok=True)
        print(f"处理事故: {accident_id}")

        # 获取摄像头信息
        camera_info = get_camera_info(camera_id)

        # 1. 剪辑事故视频
        video_filename = f"accident_{accident_id}.mp4"
        video_output = os.path.join(accident_dir, video_filename)

        # 优先使用重叠窗口视频
        if video_path and os.path.exists(video_path):
            source_video = video_path
        else:
            # 尝试找到对应的录制视频
            rec_dir = Path(RECORDINGS_DIR) / camera_id
            if rec_dir.exists():
                recordings = sorted(rec_dir.glob("seg_*.mp4"))
                source_video = str(recordings[-1]) if recordings else None
            else:
                source_video = None

        if source_video and os.path.exists(source_video):
            print(f"  剪辑视频: {source_video}")
            clip_accident_video(
                source_video, video_output,
                start_time=0,
                duration=PRE_ACCIDENT_SECONDS + POST_ACCIDENT_SECONDS
            )
        else:
            print(f"  警告: 未找到源视频")
            video_filename = "无视频"

        # 2. 提取关键帧
        keyframes = []
        if os.path.exists(video_output):
            print(f"  提取关键帧...")
            keyframes = extract_keyframes(video_output, accident_dir, num_frames=5)

        # 3. 保存原始结果 JSON
        result_json_path = os.path.join(accident_dir, "result.json")
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        # 4. 生成事故报告
        print(f"  生成报告...")
        report = generate_accident_report(
            result_data, camera_info, accident_dir,
            video_filename, keyframes
        )
        report_path = os.path.join(accident_dir, "事故分析报告.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  事故已保存: {accident_dir}")
        return True

    except Exception as e:
        print(f"处理事故失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_and_process():
    """扫描所有结果并处理事故"""
    processed = 0
    for camera_dir in Path(RESULTS_DIR).iterdir():
        if not camera_dir.is_dir():
            continue

        for result_file in camera_dir.glob("*.result.json.gz"):
            if process_accident(str(result_file)):
                processed += 1

    return processed


def watch_mode():
    """监控模式 - 持续监控新的分析结果"""
    print("事故剪辑器启动 - 监控模式")
    print(f"监控目录: {RESULTS_DIR}")
    print(f"输出目录: {ACCIDENTS_DIR}")

    processed_files = set()

    while True:
        try:
            for camera_dir in Path(RESULTS_DIR).iterdir():
                if not camera_dir.is_dir():
                    continue

                for result_file in camera_dir.glob("*.result.json.gz"):
                    file_key = str(result_file)
                    if file_key not in processed_files:
                        if process_accident(file_key):
                            print(f"新事故已处理")
                        processed_files.add(file_key)

            time.sleep(30)  # 每30秒扫描一次

        except KeyboardInterrupt:
            print("\n停止监控")
            break
        except Exception as e:
            print(f"监控出错: {e}")
            time.sleep(10)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        watch_mode()
    else:
        # 单次扫描模式
        print("扫描现有结果...")
        count = scan_and_process()
        print(f"处理完成，共处理 {count} 个事故")
