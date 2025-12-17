"""
Qwen3-VL 视频分析 Web 应用
通过阿里云 DashScope API 调用 Qwen3-VL 模型进行视频分析
"""

import sys
import io
import os

# 设置标准输出为 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 设置 HuggingFace 模型缓存目录，避免重复下载
HF_CACHE_DIR = os.path.join(os.path.dirname(__file__), "models", "huggingface")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

# 检查模型是否已缓存，如果已缓存则启用离线模式
_siglip_cache_path = os.path.join(HF_CACHE_DIR, "hub", "models--google--siglip-base-patch16-384")
if os.path.exists(_siglip_cache_path):
    os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式，不联网检查更新
    print(f"[HuggingFace] ✓ 检测到本地缓存，启用离线模式")
else:
    # 使用国内镜像加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[HuggingFace] 本地缓存不存在，使用镜像站下载模型...")
print(f"[HuggingFace] 模型缓存目录: {HF_CACHE_DIR}")

from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import os
import base64
import subprocess
import math
import re
import json
import time
import threading
from queue import Queue
from openai import OpenAI

# 新增：视频抽帧相关导入
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil
from datetime import datetime

# TrafficVLM 模块
from traffic_vlm.pipeline import TrafficVLMPipeline
from traffic_vlm.config import TrafficVLMConfig

# 可选：高级抽帧库
try:
    import decord
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False
    print("警告: 未安装decord，将使用OpenCV提取帧（速度较慢）")

try:
    from scenedetect import detect, ContentDetector
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False
    print("警告: 未安装scenedetect，场景检测功能将不可用")

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'flv', 'wmv'}
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_VIDEO_SIZE

# 上传到 API 的原始视频目标大小（为 base64 增长预留裕量）
TARGET_ORIGINAL_MB = float(os.getenv('TARGET_ORIGINAL_MB', '7.2'))
# 自适应压缩最大重试次数
MAX_COMPRESS_RETRY = int(os.getenv('MAX_COMPRESS_RETRY', '3'))

# 创建上传文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局进度队列（用于SSE推送）
progress_queues = {}

# 全局停止标志（用于中断分析）
stop_flags = {}


def is_stopped(session_id: str) -> bool:
    """检查会话是否已被请求停止"""
    return stop_flags.get(session_id, False)

# 阿里云 DashScope API 配置
# 从环境变量获取 API Key，或者直接在这里设置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-5175677ff9b4459aa45ce7ec28037515')

# 可用的模型列表
AVAILABLE_MODELS = {
    'qwen-vl-max': 'Qwen-VL-Max（最强视觉理解能力）',
    'qwen-vl-plus': 'Qwen-VL-Plus（推荐，性价比高）',
    'qwen3-vl-plus': 'Qwen3-VL-Plus（最新版本）',
    'qwen3-vl-32b-instruct': 'Qwen3-VL-32B-Instruct（阿里云部署）',
    'qwen3-vl-32b-thinking': 'Qwen3-VL-32B-Thinking（阿里云部署，思维链模式）',
    'qwen3-vl-235b-a22b-instruct': 'Qwen3-VL-235B-A22B-Instruct（阿里云部署）',
    'qwen3-vl-235b-a22b-thinking': 'Qwen3-VL-235B-A22B-Thinking（阿里云部署）'
}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_remote_model_id(model_key: str) -> str:
    """根据本地模型键获取远端模型 ID，可用环境变量覆盖。
    环境变量命名：MODEL_ID_{KEY_UPPER_UNDERSCORE}
    例如：MODEL_ID_QWEN3_VL_235B_A22B_THINKING / MODEL_ID_QWEN_VLMAX_20250813
    未设置则返回原始键值。
    """
    try:
        env_key = 'MODEL_ID_' + re.sub(r'[^A-Z0-9_]', '_', model_key.upper())
        return os.getenv(env_key, model_key)
    except Exception:
        return model_key

def encode_video_to_base64(video_path):
    """
    将视频文件编码为 base64 字符串

    Args:
        video_path: 视频文件路径

    Returns:
        str: base64 编码的视频字符串
    """
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def check_ffmpeg():
    """检查 FFmpeg 是否安装"""
    try:
        # Windows 系统上需要使用 shell=True 或者完整路径
        if sys.platform == 'win32':
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5,
                                  shell=True)
        else:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg 检测错误: {str(e)}")
        return False

def check_nvenc_support():
    """检查 FFmpeg 是否支持 NVIDIA NVENC 硬件加速"""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(['ffmpeg', '-encoders'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5,
                                  shell=True)
        else:
            result = subprocess.run(['ffmpeg', '-encoders'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)

        # 检查是否支持 h264_nvenc
        return 'h264_nvenc' in result.stdout
    except Exception as e:
        print(f"NVENC 检测错误: {str(e)}")
        return False

def compress_video(input_path, output_path, target_size_mb=6.5, session_id=None):
    """
    使用 FFmpeg 压缩视频到目标大小，并实时报告进度

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_size_mb: 目标文件大小（MB）
        session_id: 会话ID，用于进度推送

    Returns:
        bool: 是否成功
    """
    try:
        # 发送进度更新
        def send_progress(percent, message):
            if session_id and session_id in progress_queues:
                progress_queues[session_id].put({
                    'type': 'compress',
                    'progress': percent,
                    'message': message
                })

        send_progress(0, '正在分析视频...')

        # 获取视频时长
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]

        # Windows 需要 shell=True
        if sys.platform == 'win32':
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30, shell=True)
        else:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())

        print(f"视频时长: {duration:.2f} 秒")
        send_progress(10, f'视频时长: {duration:.2f}秒')

        # 计算目标码率（考虑音频码率约128kbps）
        target_bitrate = int((target_size_mb * 8 * 1024) / duration - 128)
        if target_bitrate < 100:
            target_bitrate = 100  # 最低100kbps

        print(f"目标视频码率: {target_bitrate} kbps")
        send_progress(15, '开始压缩视频...')

        # 模拟进度更新
        import threading

        # 获取输入文件大小，用于更准确的进度估算
        input_size_mb = os.path.getsize(input_path) / (1024 * 1024)

        # 动态设定分辨率/帧率策略（随目标码率降低）
        # 最小宽度854，避免过度压缩损失细节
        scale_width = None
        output_fps = None
        if target_bitrate < 220:
            scale_width = 854  # 最小宽度854
            output_fps = 15
        elif target_bitrate < 350:
            scale_width = 854  # 最小宽度854
            output_fps = 20
        elif target_bitrate < 600:
            scale_width = 854
        elif target_bitrate < 1000:
            scale_width = 1280

        # 检测是否支持 NVIDIA GPU 加速
        use_gpu = check_nvenc_support()

        if use_gpu:
            print("✓ 检测到 NVIDIA GPU 支持，使用硬件加速")
            send_progress(15, '使用 GPU 硬件加速压缩...')

            # GPU 加速命令（使用 NVENC）
            compress_cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',                    # 硬件加速
                '-hwaccel_output_format', 'cuda',      # 保持数据在GPU显存
                '-i', input_path,
                '-c:v', 'h264_nvenc',                  # NVIDIA H.264 编码器
                '-b:v', f'{target_bitrate}k',
                '-maxrate', f'{target_bitrate}k',
                '-bufsize', f'{target_bitrate * 2}k',
                '-preset', 'p4',                       # p4 = medium (p1最快/质量最低, p7最慢/质量最高)
                '-rc', 'vbr',                          # 可变码率
                '-vf', 'scale_cuda=trunc(iw/2)*2:trunc(ih/2)*2',  # GPU缩放滤镜
                '-c:a', 'aac',                         # 音频编码
                '-b:a', '128k',
                '-y',
                output_path
            ]
            # 动态过滤器（覆盖默认 -vf），在 GPU 路径下也允许使用 CPU 过滤器以达到更小体积
            if 'compress_cmd' in locals():
                if scale_width or output_fps:
                    _filters = []
                    if scale_width:
                        _filters.append(f'scale={scale_width}:-2')
                    if output_fps:
                        _filters.append(f'fps={output_fps}')
                    try:
                        compress_cmd.insert(-1, '-filter:v')
                        compress_cmd.insert(-1, ','.join(_filters))
                    except Exception:
                        pass

            # GPU 加速大约每秒处理 2-5 秒的视频内容（比CPU快3-10倍）
            estimated_time = duration / 2.0  # 秒
        else:
            print("✗ 未检测到 NVIDIA GPU 支持，使用 CPU 编码")
            send_progress(15, '使用 CPU 压缩（较慢）...')

            # CPU 编码命令
            compress_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',                     # CPU H.264 编码器
                '-b:v', f'{target_bitrate}k',
                '-maxrate', f'{target_bitrate}k',
                '-bufsize', f'{target_bitrate * 2}k',
                '-preset', 'medium',                   # CPU preset
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y',
                output_path
            ]
            # 动态过滤器（覆盖默认 -vf）
            if 'compress_cmd' in locals():
                if scale_width or output_fps:
                    _filters = []
                    if scale_width:
                        _filters.append(f'scale={scale_width}:-2')
                    if output_fps:
                        _filters.append(f'fps={output_fps}')
                    try:
                        compress_cmd.insert(-1, '-filter:v')
                        compress_cmd.insert(-1, ','.join(_filters))
                    except Exception:
                        pass

            # CPU 大约每秒处理 0.3 秒的视频内容
            estimated_time = duration / 0.3  # 秒

        print(f"预估压缩时间: {estimated_time:.1f} 秒 ({estimated_time/60:.1f} 分钟)")
        print(f"执行压缩命令: {' '.join(compress_cmd)}")

        # 启动压缩进程 - 关键修复：stderr 重定向到 DEVNULL 避免缓冲区阻塞
        if sys.platform == 'win32':
            process = subprocess.Popen(compress_cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,  # 避免 stderr 缓冲区阻塞
                                      shell=True)
        else:
            process = subprocess.Popen(compress_cmd,
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

        # 模拟进度更新线程 - 使用更准确的估算
        def simulate_progress():
            start_time = time.time()
            last_progress = 15

            while process.poll() is None:  # 进程还在运行
                elapsed = time.time() - start_time

                # 基于预估时间计算进度（15% -> 95%）
                if estimated_time > 0:
                    progress_ratio = elapsed / estimated_time
                    estimated_progress = min(95, 15 + int(progress_ratio * 80))
                else:
                    # 降级方案：基于视频时长
                    estimated_progress = min(95, 15 + int((elapsed / (duration * 0.5)) * 80))

                # 确保进度只增不减
                estimated_progress = max(last_progress, estimated_progress)
                last_progress = estimated_progress

                # 格式化消息
                if elapsed < 60:
                    time_msg = f'压缩中... {int(elapsed)}s'
                else:
                    minutes = int(elapsed / 60)
                    seconds = int(elapsed % 60)
                    time_msg = f'压缩中... {minutes}m {seconds}s'

                # 如果超过预估时间，显示提示
                if elapsed > estimated_time * 1.2:
                    time_msg += ' (即将完成...)'

                send_progress(estimated_progress, time_msg)
                time.sleep(2)  # 每2秒更新一次

        progress_thread = threading.Thread(target=simulate_progress)
        progress_thread.daemon = True
        progress_thread.start()

        # 等待进程完成
        returncode = process.wait()

        # 等待进度线程结束
        progress_thread.join(timeout=1)

        # 检查结果
        if returncode == 0 and os.path.exists(output_path):
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"压缩成功！输出文件大小: {output_size:.2f} MB")
            if use_gpu:
                send_progress(100, f'GPU加速压缩完成！文件大小: {output_size:.2f}MB')
            else:
                send_progress(100, f'压缩完成！文件大小: {output_size:.2f}MB')
            return True
        else:
            print(f"FFmpeg 返回码: {returncode}")

            # 如果 GPU 加速失败，自动降级到 CPU 编码
            if use_gpu and returncode != 0:
                print("⚠ GPU 加速失败，自动切换到 CPU 编码...")
                send_progress(15, 'GPU失败，切换到CPU编码...')

                # CPU 编码命令
                cpu_compress_cmd = [
                    'ffmpeg',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-b:v', f'{target_bitrate}k',
                    '-maxrate', f'{target_bitrate}k',
                    '-bufsize', f'{target_bitrate * 2}k',
                    '-preset', 'medium',
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-y',
                    output_path
                ]

                print(f"使用 CPU 编码重试...")
                print(f"执行命令: {' '.join(cpu_compress_cmd)}")

                # 重新估算时间（CPU 较慢）
                estimated_time = duration / 0.3

                # 启动 CPU 编码进程
                if sys.platform == 'win32':
                    cpu_process = subprocess.Popen(cpu_compress_cmd,
                                                   stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.DEVNULL,
                                                   shell=True)
                else:
                    cpu_process = subprocess.Popen(cpu_compress_cmd,
                                                   stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.DEVNULL)

                # 重新启动进度线程
                def simulate_cpu_progress():
                    start_time = time.time()
                    last_progress = 15

                    while cpu_process.poll() is None:
                        elapsed = time.time() - start_time

                        if estimated_time > 0:
                            progress_ratio = elapsed / estimated_time
                            estimated_progress = min(95, 15 + int(progress_ratio * 80))
                        else:
                            estimated_progress = min(95, 15 + int((elapsed / (duration * 0.5)) * 80))

                        estimated_progress = max(last_progress, estimated_progress)
                        last_progress = estimated_progress

                        if elapsed < 60:
                            time_msg = f'CPU压缩中... {int(elapsed)}s'
                        else:
                            minutes = int(elapsed / 60)
                            seconds = int(elapsed % 60)
                            time_msg = f'CPU压缩中... {minutes}m {seconds}s'

                        if elapsed > estimated_time * 1.2:
                            time_msg += ' (即将完成...)'

                        send_progress(estimated_progress, time_msg)
                        time.sleep(2)

                cpu_progress_thread = threading.Thread(target=simulate_cpu_progress)
                cpu_progress_thread.daemon = True
                cpu_progress_thread.start()

                # 等待 CPU 编码完成
                cpu_returncode = cpu_process.wait()
                cpu_progress_thread.join(timeout=1)

                if cpu_returncode == 0 and os.path.exists(output_path):
                    output_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"CPU 编码成功！输出文件大小: {output_size:.2f} MB")
                    send_progress(100, f'CPU编码完成！文件大小: {output_size:.2f}MB')
                    return True
                else:
                    print(f"CPU 编码也失败了，返回码: {cpu_returncode}")
                    if not os.path.exists(output_path):
                        print("错误：输出文件不存在")
                    send_progress(0, '压缩失败')
                    return False
            else:
                if not os.path.exists(output_path):
                    print("错误：输出文件不存在")
                send_progress(0, '压缩失败')
                return False

    except Exception as e:
        print(f"压缩视频时出错: {str(e)}")
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'compress',
                'progress': 0,
                'message': f'压缩失败: {str(e)}'
            })
        return False

# ============================================================
# 视频抽帧模块
# ============================================================

def extract_frames_uniform(video_path, fps=1.0, max_frames=None, session_id=None):
    """
    均匀采样：按固定FPS提取帧

    Args:
        video_path: 视频文件路径
        fps: 采样帧率（每秒提取几帧）
        max_frames: 最大帧数限制
        session_id: 会话ID，用于进度推送

    Returns:
        list: 提取的帧列表（PIL Image对象）
        dict: 元数据（时间戳、帧号等）
    """
    def send_progress(percent, message):
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'sampling',
                'progress': percent,
                'message': message
            })

    try:
        send_progress(0, '开始提取帧...')

        # 使用OpenCV打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        # 获取视频信息
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps

        print(f"视频信息: FPS={video_fps}, 总帧数={total_frames}, 时长={duration:.2f}秒")
        send_progress(10, f'视频时长{duration:.2f}秒，开始提取...')

        # 计算采样间隔
        frame_interval = int(video_fps / fps)

        frames = []
        metadata = {
            'timestamps': [],
            'frame_indices': [],
            'video_fps': video_fps,
            'total_frames': total_frames,
            'duration': duration
        }

        frame_count = 0
        extracted_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                frames.append(pil_image)
                metadata['timestamps'].append(frame_count / video_fps)
                metadata['frame_indices'].append(frame_count)

                extracted_count += 1

                # 更新进度
                progress = int(10 + (frame_count / total_frames) * 80)
                send_progress(progress, f'已提取 {extracted_count} 帧...')

                # 检查是否达到最大帧数
                if max_frames and extracted_count >= max_frames:
                    break

            frame_count += 1

        cap.release()

        print(f"提取完成：共提取 {len(frames)} 帧")
        send_progress(100, f'提取完成，共{len(frames)}帧')

        return frames, metadata

    except Exception as e:
        print(f"提取帧失败: {str(e)}")
        send_progress(0, f'提取失败: {str(e)}')
        raise


def extract_frames_keyframes(video_path, num_keyframes=16, session_id=None):
    """
    关键帧提取：均匀提取N个关键帧

    Args:
        video_path: 视频文件路径
        num_keyframes: 提取的关键帧数量
        session_id: 会话ID

    Returns:
        list: 提取的帧列表
        dict: 元数据
    """
    def send_progress(percent, message):
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'sampling',
                'progress': percent,
                'message': message
            })

    try:
        send_progress(0, '开始提取关键帧...')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # 计算均匀间隔
        indices = np.linspace(0, total_frames - 1, num_keyframes, dtype=int)

        frames = []
        metadata = {
            'timestamps': [],
            'frame_indices': [],
            'video_fps': video_fps,
            'total_frames': total_frames
        }

        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                frames.append(pil_image)
                metadata['timestamps'].append(frame_idx / video_fps)
                metadata['frame_indices'].append(int(frame_idx))

                progress = int(10 + (i / len(indices)) * 90)
                send_progress(progress, f'提取关键帧 {i+1}/{num_keyframes}...')

        cap.release()

        print(f"关键帧提取完成：{len(frames)} 帧")
        send_progress(100, f'提取完成，共{len(frames)}个关键帧')

        return frames, metadata

    except Exception as e:
        print(f"提取关键帧失败: {str(e)}")
        send_progress(0, f'提取失败: {str(e)}')
        raise


def extract_frames_accident_analysis(video_path, config, session_id=None):
    """
    交通事故分析专用抽帧策略
    实现四阶段采样（简化版）

    Args:
        video_path: 视频文件路径
        config: 配置字典
        session_id: 会话ID

    Returns:
        list: 提取的帧列表（按阶段分组）
        dict: 元数据（包含各阶段信息）
    """
    def send_progress(percent, message):
        if session_id and session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'sampling',
                'progress': percent,
                'message': message
            })

    try:
        send_progress(0, '交通事故分析：阶段1 - 粗扫描...')

        # 阶段1：粗粒度扫描（1 FPS）
        frames_stage1, meta1 = extract_frames_uniform(
            video_path,
            fps=1.0,
            max_frames=600,  # 最多10分钟
            session_id=None  # 不重复推送进度
        )

        send_progress(25, f'阶段1完成：扫描{len(frames_stage1)}帧')

        # 阶段2：选择关键时间段的帧（模拟事故时刻检测）
        # 这里简化为选择视频中段的高密度采样
        duration = meta1['duration']
        accident_time = duration / 2  # 假设事故在中间

        send_progress(50, '阶段2：精确定位事故时刻...')

        # 提取事故时刻附近的密集帧（±10秒）
        frames_stage2 = []
        for i, ts in enumerate(meta1['timestamps']):
            if abs(ts - accident_time) <= 10:  # 事故前后10秒
                frames_stage2.append(frames_stage1[i])

        send_progress(75, f'阶段2完成：定位到{len(frames_stage2)}帧')

        # 阶段3：环境分析关键帧（选择5个代表帧）
        num_env_frames = min(5, len(frames_stage1))
        env_indices = np.linspace(0, len(frames_stage1)-1, num_env_frames, dtype=int)
        frames_stage3 = [frames_stage1[i] for i in env_indices]

        send_progress(90, '阶段3：提取环境分析帧...')

        # 合并所有帧（简单合并，不去重，因为PIL Image对象不可哈希）
        # 由于frames_stage2和frames_stage3是从frames_stage1中选取的，重复影响不大
        all_frames = frames_stage1 + frames_stage2 + frames_stage3

        metadata = {
            'strategy': 'accident_analysis',
            'total_frames': len(all_frames),
            'stage1_frames': len(frames_stage1),
            'stage2_frames': len(frames_stage2),
            'stage3_frames': len(frames_stage3),
            'video_duration': duration,
            'estimated_accident_time': accident_time,
            'config': config
        }

        send_progress(100, f'事故分析完成：共{len(all_frames)}帧')

        return all_frames, metadata

    except Exception as e:
        print(f"事故分析抽帧失败: {str(e)}")
        send_progress(0, f'抽帧失败: {str(e)}')
        raise


def save_frames_to_folder(frames, strategy_name, video_name=None):
    """
    保存抽取的帧到文件夹（用于调试）

    Args:
        frames: PIL Image列表
        strategy_name: 抽帧策略名称
        video_name: 视频文件名（可选）

    Returns:
        str: 保存的文件夹路径
    """
    from datetime import datetime

    # 创建时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 创建文件夹名称：策略名_时间戳
    folder_name = f"{strategy_name}_{timestamp}"
    if video_name:
        # 移除文件扩展名
        video_base = os.path.splitext(video_name)[0]
        folder_name = f"{video_base}_{strategy_name}_{timestamp}"

    # 创建保存路径
    debug_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_frames')
    os.makedirs(debug_folder, exist_ok=True)

    save_path = os.path.join(debug_folder, folder_name)
    os.makedirs(save_path, exist_ok=True)

    # 保存所有帧
    print(f"\n💾 保存抽帧图片到: {save_path}")
    for i, frame in enumerate(frames):
        frame_path = os.path.join(save_path, f"frame_{i:04d}.jpg")
        frame.save(frame_path, format='JPEG', quality=95)

    print(f"✅ 已保存 {len(frames)} 帧到文件夹: {folder_name}\n")
    return save_path


def calculate_image_tokens(width, height):
    """
    计算图片的Token数量（Qwen3-VL规则：每32x32像素=1 Token）

    Args:
        width: 图片宽度
        height: 图片高度

    Returns:
        int: Token数量
    """
    import math
    # Qwen3-VL: 每32x32像素对应1个Token
    tokens = math.ceil(width / 32) * math.ceil(height / 32)
    # 最少4个Token，最多16384个Token
    return max(4, min(tokens, 16384))


def frames_to_base64_images(frames, max_tokens=250000):
    """
    将帧列表转换为base64编码的图片列表（用于API调用）
    使用Token限制而非文件大小限制，确保符合Qwen-VL API要求

    Args:
        frames: PIL Image列表
        max_tokens: 最大Token数（默认250000，为258048留余量）

    Returns:
        list: base64编码的图片URL列表
    """
    import io

    base64_images = []
    total_tokens = 0
    total_size_mb = 0

    print(f"\n🔢 开始转换帧，使用Token限制策略（最大{max_tokens} tokens）")

    for i, frame in enumerate(frames):
        # 压缩图片：降低分辨率以减少Token消耗
        # 最大宽度1280px，这样可以在保持质量的同时减少Token
        max_width = 1280
        width, height = frame.size
        if width > max_width:
            ratio = max_width / width
            new_size = (max_width, int(height * ratio))
            frame = frame.resize(new_size, Image.Resampling.LANCZOS)
            width, height = new_size

        # 计算这张图片需要的Token数
        frame_tokens = calculate_image_tokens(width, height)

        # 检查Token限制
        if total_tokens + frame_tokens > max_tokens:
            print(f"⚠️ 已达到Token限制（{total_tokens}/{max_tokens}），停止添加更多帧")
            print(f"   成功处理 {i}/{len(frames)} 帧")
            break

        # 转换为JPEG并压缩
        buffer = io.BytesIO()
        frame.save(buffer, format='JPEG', quality=85, optimize=True)  # 提高质量，保留更多细节
        img_bytes = buffer.getvalue()

        # 编码为base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        img_url = f"data:image/jpeg;base64,{img_base64}"

        # 记录统计信息
        img_size_mb = len(img_base64) / (1024 * 1024)
        total_size_mb += img_size_mb
        total_tokens += frame_tokens

        base64_images.append(img_url)

        # 每10帧打印一次进度
        if (i + 1) % 10 == 0:
            print(f"   进度: {i+1}/{len(frames)} 帧，Token: {total_tokens}/{max_tokens}，大小: {total_size_mb:.2f}MB")

    print(f"✅ 转换完成：{len(base64_images)}/{len(frames)} 帧")
    print(f"   总Token数: {total_tokens} ({total_tokens/max_tokens*100:.1f}%)")
    print(f"   总大小: {total_size_mb:.2f}MB")
    print(f"   平均每帧: {total_tokens/len(base64_images):.0f} tokens, {total_size_mb/len(base64_images):.2f}MB\n")

    return base64_images


def analyze_video_with_api(video_path=None, video_url=None, prompt='请详细描述这个视频中发生了什么。', model='qwen-vl-plus', session_id=None):
    """
    使用阿里云 DashScope API 分析视频内容

    Args:
        video_path: 视频文件路径（本地上传方式）
        video_url: 视频URL（URL方式，支持最大2GB）
        prompt: 用户提问
        model: 使用的模型名称
        session_id: 会话ID，用于进度推送

    Returns:
        str: 模型分析结果
    """
    try:
        # 发送进度更新
        def send_progress(percent, message):
            if session_id and session_id in progress_queues:
                progress_queues[session_id].put({
                    'type': 'upload',
                    'progress': percent,
                    'message': message
                })
        # 检查 API Key
        if not DASHSCOPE_API_KEY:
            raise ValueError(
                "未设置 DASHSCOPE_API_KEY！\n"
                "请在环境变量中设置 API Key，或在 app.py 中直接配置。\n"
                "获取 API Key: https://dashscope.console.aliyun.com/apiKey"
            )

        send_progress(0, '开始分析视频...')

        remote_model = get_remote_model_id(model)
        print(f"使用模型: {model} -> {remote_model}")
        print(f"用户提问: {prompt}")

        # 判断使用URL方式还是本地文件方式
        if video_url:
            # URL方式 - 支持最大2GB视频
            print(f"使用URL方式分析视频: {video_url}")
            send_progress(10, '使用URL方式，准备调用API...')

            # 构建请求消息 - URL方式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": video_url
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ]

        else:
            # 本地文件方式 - base64编码
            print(f"开始分析视频: {video_path}")

            # 检查视频文件大小
            video_size = os.path.getsize(video_path)
            video_size_mb = video_size / (1024 * 1024)
            print(f"视频文件大小: {video_size_mb:.2f} MB")

            send_progress(5, f'准备上传视频 ({video_size_mb:.2f}MB)...')

            # API限制：base64编码后不能超过10MB
            # 由于base64编码会增加约33%的大小，原始文件应该小于7.5MB
            max_original_size = 7.5 * 1024 * 1024  # 7.5MB
            # 使用可配置的阈值覆盖默认 7.5MB（为 base64 增长预留）
            try:
                max_original_size = TARGET_ORIGINAL_MB * 1024 * 1024
            except Exception:
                pass
            if video_size > max_original_size:
                raise ValueError(
                    f"视频文件太大（{video_size_mb:.2f} MB）！\n"
                    f"DashScope API 限制 base64 编码后的视频不能超过 10MB。\n"
                    f"建议：\n"
                    f"1. 使用视频压缩工具压缩视频\n"
                    f"2. 截取较短的视频片段（建议 < 7MB）\n"
                    f"3. 降低视频分辨率或帧率"
                )

            # 编码视频为 base64
            print("正在编码视频...")
            send_progress(10, '正在编码视频...')
            base64_video = encode_video_to_base64(video_path)
            base64_size_mb = len(base64_video) / (1024 * 1024)
            print(f"视频编码完成，base64 大小: {base64_size_mb:.2f} MB")
            send_progress(30, f'视频编码完成 ({base64_size_mb:.2f}MB)')

            # 再次检查编码后的大小
            if len(base64_video) > 10 * 1024 * 1024:
                raise ValueError(
                    f"编码后视频太大（{base64_size_mb:.2f} MB），超过 API 限制（10MB）！\n"
                    f"请压缩视频或使用更短的片段。"
                )

            # 构建请求消息 - base64方式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"data:video/mp4;base64,{base64_video}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ]
                }
            ]

        # 创建 OpenAI 客户端（兼容模式）
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 调用 API
        print("正在调用 DashScope API...")
        send_progress(40, '正在上传到AI模型...')

        completion = client.chat.completions.create(
            model=remote_model,
            messages=messages,
        )

        send_progress(90, 'AI正在分析视频...')

        # 获取结果
        result = completion.choices[0].message.content
        print("分析完成！")
        send_progress(100, '分析完成！')

        return result

    except Exception as e:
        import traceback
        error_msg = f"API 调用失败: {str(e)}"
        print(error_msg)
        print("详细错误信息:")
        print(traceback.format_exc())
        raise Exception(error_msg)


def analyze_images_with_api(base64_images, prompt='请分析这些图片。', model='qwen-vl-plus', session_id=None):
    """
    使用阿里云 DashScope API 分析多张图片

    Args:
        base64_images: base64编码的图片URL列表
        prompt: 用户提问
        model: 使用的模型名称
        session_id: 会话ID

    Returns:
        str: 模型分析结果
    """
    try:
        def send_progress(percent, message):
            if session_id and session_id in progress_queues:
                progress_queues[session_id].put({
                    'type': 'upload',
                    'progress': percent,
                    'message': message
                })

        send_progress(0, '准备发送图片到AI模型...')

        # 检查 API Key
        if not DASHSCOPE_API_KEY:
            raise ValueError("未设置 DASHSCOPE_API_KEY！")

        remote_model = get_remote_model_id(model)
        print(f"使用模型: {model} -> {remote_model}")
        print(f"分析图片数量: {len(base64_images)}")

        # 构建消息内容（多图）
        content = []
        for img_url in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

        # 添加文本提问
        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        # 创建 OpenAI 客户端
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        send_progress(40, f'上传{len(base64_images)}张图片到AI模型...')

        # 调用 API
        completion = client.chat.completions.create(
            model=remote_model,
            messages=messages,
        )

        send_progress(90, 'AI正在分析图片...')

        # 获取结果
        result = completion.choices[0].message.content
        print("多图分析完成！")
        send_progress(100, '分析完成！')

        return result

    except Exception as e:
        import traceback
        error_msg = f"多图API调用失败: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        raise Exception(error_msg)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/progress/<session_id>')
def progress(session_id):
    """SSE 进度推送端点"""
    def generate():
        # 创建该会话的进度队列
        if session_id not in progress_queues:
            progress_queues[session_id] = Queue()

        q = progress_queues[session_id]

        # 发送初始连接消息
        yield f"data: {json.dumps({'type': 'connected', 'message': '已连接'})}\n\n"

        try:
            while True:
                try:
                    # 从队列获取进度更新，设置较短的超时以便发送心跳
                    data = q.get(timeout=5)  # 5秒超时

                    # 发送 SSE 数据
                    yield f"data: {json.dumps(data)}\n\n"

                    # 如果任务完成或失败，结束流
                    if data.get('type') == 'complete' or data.get('type') == 'error':
                        break

                except:
                    # 队列超时，发送心跳保持连接
                    yield f": heartbeat\n\n"
                    continue

        except GeneratorExit:
            # 客户端断开连接
            pass
        finally:
            # 清理队列
            if session_id in progress_queues:
                del progress_queues[session_id]

    return Response(stream_with_context(generate()),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'X-Accel-Buffering': 'no',
                       'Connection': 'keep-alive'
                   })

@app.route('/analyze', methods=['POST'])
def analyze():
    """处理视频分析请求 - 立即返回session_id，后台处理"""
    try:
        # 获取输入方式
        input_method = request.form.get('input_method', 'upload')  # 'upload' 或 'url'

        # 获取用户提问和模型选择
        prompt = request.form.get('prompt', '请详细描述这个视频中发生了什么。')
        model = request.form.get('model', 'qwen-vl-plus')
        analysis_mode = request.form.get('analysis_mode', 'traffic_vlm')
        event_query = request.form.get('event_query', '').strip()
        camera_id = request.form.get('camera_id', 'camera-1').strip() or 'camera-1'

        # 根据输入方式处理
        video_url = None
        filepath = None
        filename = None
        auto_compress = False

        if input_method == 'url':
            # URL方式
            video_url = request.form.get('video_url', '').strip()
            if not video_url:
                return jsonify({'error': '请提供视频URL'}), 400

            # 验证URL格式
            if not video_url.startswith(('http://', 'https://')):
                return jsonify({'error': '请提供有效的HTTP/HTTPS视频URL'}), 400

            print(f"收到URL方式请求: {video_url}")

        else:
            # 上传文件方式
            # 检查是否有文件
            if 'video' not in request.files:
                return jsonify({'error': '未找到视频文件'}), 400

            file = request.files['video']

            # 检查文件名
            if file.filename == '':
                return jsonify({'error': '未选择文件'}), 400

            # 检查文件类型
            if not allowed_file(file.filename):
                return jsonify({'error': f'不支持的文件格式，请上传: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

            auto_compress = request.form.get('auto_compress', 'true').lower() == 'true'

            # 保存文件
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # 验证模型选择
        if model not in AVAILABLE_MODELS:
            # 允许额外的模型键（未在 AVAILABLE_MODELS 中展示）
            extra_allowed = {'qwen3-vl-235b-a22b-thinking', 'qwen-vlmax-20250813', 'qwen3-vl-32b-instruct', 'qwen3-vl-32b-thinking'}
            if model not in extra_allowed:
                model = 'qwen-vl-plus'

        # ========== 提前获取所有请求参数（避免在线程中访问request） ==========
        # 获取抽帧策略相关参数
        sampling_strategy = request.form.get('sampling_strategy', 'full_video')
        uniform_fps = float(request.form.get('uniform_fps', 1.0)) if sampling_strategy == 'uniform_fps' else 1.0
        keyframe_count = int(request.form.get('keyframe_count', 16)) if sampling_strategy == 'keyframe_only' else 16

        # 交通事故分析配置
        accident_config = None
        if sampling_strategy == 'accident_analysis':
            accident_config = {
                'detect_accident_time': request.form.get('detect_accident_time', 'true').lower() == 'true',
                'track_trajectory': request.form.get('track_trajectory', 'true').lower() == 'true',
                'analyze_environment': request.form.get('analyze_environment', 'true').lower() == 'true'
            }

        # 生成唯一的会话ID
        session_id = f"{int(time.time() * 1000)}_{os.getpid()}"

        # 创建进度队列
        progress_queues[session_id] = Queue()

        # 在后台线程中处理任务
        def process_video():
            compressed_path = None
            final_video_path = filepath
            try:
                # 检查是否已被停止
                if is_stopped(session_id):
                    print(f"[Session {session_id}] 任务已被停止（启动前）")
                    return

                user_intent = event_query or prompt
                video_source_path = video_url if video_url else filepath

                def pipeline_progress(percent, message):
                    # 检查是否已被停止
                    if is_stopped(session_id):
                        raise InterruptedError("分析已被用户停止")
                    progress_queues[session_id].put({
                        'type': 'analysis',
                        'progress': int(percent),
                        'message': message
                    })

                def format_pipeline_summary(res):
                    lines = []
                    lines.append(f"Keyframes: {len(res.get('keyframes', []))}")
                    lines.append(f"Candidate clips: {len(res.get('clips', []))}")
                    for item in res.get('results', []):
                        clip = item.get('clip', {})
                        vlm_out = item.get('vlm_output', {}) or {}
                        vio = vlm_out.get('violations') or []
                        vio_text = '; '.join([f"{v.get('type')}({v.get('confidence', 0):.2f})" for v in vio]) if vio else 'No high-confidence violations'
                        lines.append(f"- {clip.get('clip_id', '')} [{clip.get('start_time', 0):.1f}-{clip.get('end_time', 0):.1f}s] score {clip.get('clip_score', 0):.3f} | {vio_text}")
                    return "\n".join(lines)

                def extract_detailed_analysis(res):
                    """提取详细的分析结果（包含大模型的文本描述）"""
                    results = res.get('results', [])
                    if not results:
                        return "未检测到相关内容"

                    analysis_parts = []
                    for i, item in enumerate(results, 1):
                        vlm_out = item.get('vlm_output', {}) or {}
                        text_summary = vlm_out.get('text_summary', '无描述')
                        has_violation = vlm_out.get('has_violation', False)
                        violations = vlm_out.get('violations', [])

                        # 添加片段信息
                        clip = item.get('clip', {})
                        start_time = clip.get('start_time', 0)
                        end_time = clip.get('end_time', 0)

                        analysis_parts.append(f"【片段 {i}】时间: {start_time:.1f}s - {end_time:.1f}s")
                        analysis_parts.append(f"分析结果: {text_summary}")

                        # 添加违法信息
                        if violations:
                            analysis_parts.append("检测到的违法行为:")
                            for v in violations:
                                vtype = v.get('type', '')
                                confidence = v.get('confidence', 0)
                                evidence = v.get('evidence', '')
                                analysis_parts.append(f"  - {vtype} (置信度: {confidence:.2f})")
                                if evidence:
                                    analysis_parts.append(f"    依据: {evidence}")
                        elif has_violation:
                            analysis_parts.append("检测到违法行为，但未能识别具体类型")
                        else:
                            analysis_parts.append("未检测到违法行为")

                        analysis_parts.append("")  # 空行分隔

                    return "\n".join(analysis_parts)

                if analysis_mode == 'traffic_vlm':
                    # 检查是否已被停止
                    if is_stopped(session_id):
                        print(f"[Session {session_id}] 任务已被停止（pipeline启动前）")
                        return

                    print(f"\n{'='*60}")
                    print(f"TrafficVLM pipeline start (Session: {session_id})")
                    print(f"Source: {video_source_path}")
                    print(f"Camera ID: {camera_id}")
                    print(f"Query: {user_intent}")
                    print(f"{'='*60}\n")

                    try:
                        pipeline = TrafficVLMPipeline(config=TrafficVLMConfig(), progress_cb=pipeline_progress)
                        pipeline_result = pipeline.run(video_source_path, user_intent, camera_id=camera_id, mode="violation")

                        # 返回详细的分析结果，而不是摘要
                        detailed_analysis = extract_detailed_analysis(pipeline_result)

                        response_data = {
                            'type': 'complete',
                            'success': True,
                            'result': detailed_analysis,
                            'analysis_mode': 'traffic_vlm',
                            'video_source': video_source_path,
                            'model': model,
                            'pipeline': pipeline_result
                        }
                        progress_queues[session_id].put(response_data)
                    except InterruptedError as e:
                        print(f"[Session {session_id}] TrafficVLM pipeline 被用户停止")
                        # 不发送错误消息，停止消息已通过 /stop 端点发送
                    except Exception as e:
                        progress_queues[session_id].put({
                            'type': 'error',
                            'message': f'TrafficVLM pipeline failed: {str(e)}'
                        })
                    return

                if analysis_mode == 'accident_search':
                    # 检查是否已被停止
                    if is_stopped(session_id):
                        print(f"[Session {session_id}] 任务已被停止（accident pipeline启动前）")
                        return

                    print(f"\n{'='*60}")
                    print(f"Accident Search pipeline start (Session: {session_id})")
                    print(f"Source: {video_source_path}")
                    print(f"Camera ID: {camera_id}")
                    print(f"Query: {user_intent}")
                    print(f"{'='*60}\n")

                    try:
                        pipeline = TrafficVLMPipeline(config=TrafficVLMConfig(), progress_cb=pipeline_progress)
                        pipeline_result = pipeline.run(video_source_path, user_intent, camera_id=camera_id, mode="accident")

                        # 返回详细的分析结果
                        detailed_analysis = extract_detailed_analysis(pipeline_result)

                        response_data = {
                            'type': 'complete',
                            'success': True,
                            'result': detailed_analysis,
                            'analysis_mode': 'accident_search',
                            'video_source': video_source_path,
                            'model': model,
                            'pipeline': pipeline_result
                        }
                        progress_queues[session_id].put(response_data)
                    except InterruptedError as e:
                        print(f"[Session {session_id}] Accident Search pipeline 被用户停止")
                        # 不发送错误消息，停止消息已通过 /stop 端点发送
                    except Exception as e:
                        progress_queues[session_id].put({
                            'type': 'error',
                            'message': f'Accident Search pipeline failed: {str(e)}'
                        })
                    return

                if video_url:
                    print(f"\n{'='*60}")
                    print(f"Received video analysis request (Session: {session_id})")
                    print(f"Video URL: {video_url}")
                    print(f"Input method: URL (max 10GB)")
                    print(f"{'='*60}\n")

                    result = analyze_video_with_api(video_url=video_url, prompt=prompt, model=model, session_id=session_id)

                    response_data = {
                        'type': 'complete',
                        'success': True,
                        'result': result,
                        'video_source': video_url,
                        'model': model,
                        'input_method': 'url'
                    }

                    progress_queues[session_id].put(response_data)

                else:
                    file_size_mb = os.path.getsize(filepath) / (1024*1024)

                    print(f"\n{'='*60}")
                    print(f"Received video analysis request (Session: {session_id})")
                    print(f"Video file: {filename}")
                    print(f"Path: {filepath}")
                    print(f"Size: {file_size_mb:.2f} MB")
                    print(f"Auto compress: {auto_compress}")
                    print(f"Sampling: {sampling_strategy}")
                    print(f"{'='*60}\n")

                    final_video_path = filepath
                    compressed_filename = None

                    if auto_compress and file_size_mb > 7.0 and sampling_strategy == 'full_video':
                        print("Large file, start auto compression...")

                        if not check_ffmpeg():
                            progress_queues[session_id].put({
                                'type': 'error',
                                'message': 'FFmpeg not installed, cannot compress.'
                            })
                            return

                        name, ext = os.path.splitext(filename)
                        compressed_filename = f"{name}_compressed{ext}"
                        compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], compressed_filename)

                        target_size_mb = 6.5
                        success = compress_video(filepath, compressed_path, target_size_mb=target_size_mb, session_id=session_id)

                        if success:
                            print(f"Use compressed video: {compressed_filename}")
                            final_video_path = compressed_path
                            try:
                                current_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                            except Exception:
                                current_size_mb = None

                            retry = 0
                            while current_size_mb is not None and current_size_mb > TARGET_ORIGINAL_MB and retry < MAX_COMPRESS_RETRY:
                                retry += 1
                                ratio = TARGET_ORIGINAL_MB / max(current_size_mb, 0.01)
                                new_target = max(1.0, target_size_mb * ratio * 0.9)

                                print(f"Compressed still {current_size_mb:.2f} MB, retry {retry}, target {new_target:.2f} MB")
                                progress_queues[session_id].put({
                                    'type': 'compress',
                                    'progress': 15,
                                    'message': f'Re-compress #{retry}, target {new_target:.2f}MB'
                                })

                                target_size_mb = new_target
                                success = compress_video(filepath, compressed_path, target_size_mb=target_size_mb, session_id=session_id)
                                if not success:
                                    print("Adaptive compression failed, keep last result")
                                    break

                                try:
                                    current_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                                except Exception:
                                    current_size_mb = None

                            if current_size_mb is not None and current_size_mb > TARGET_ORIGINAL_MB:
                                progress_queues[session_id].put({
                                    'type': 'compress',
                                    'progress': 95,
                                    'message': f'Still above threshold ({current_size_mb:.2f}MB>{TARGET_ORIGINAL_MB:.2f}MB), may fail upload'
                                })
                        else:
                            print("Compression failed, use original video")

                    if sampling_strategy != 'full_video':
                        print(f"\nSampling strategy: {sampling_strategy}")

                        if sampling_strategy == 'uniform_fps':
                            frames, metadata = extract_frames_uniform(final_video_path, fps=uniform_fps, session_id=session_id)
                        elif sampling_strategy == 'keyframe_only':
                            frames, metadata = extract_frames_keyframes(final_video_path, num_keyframes=keyframe_count, session_id=session_id)
                        elif sampling_strategy == 'accident_analysis':
                            frames, metadata = extract_frames_accident_analysis(final_video_path, accident_config, session_id=session_id)
                        else:
                            frames, metadata = extract_frames_uniform(final_video_path, fps=1.0, session_id=session_id)

                        try:
                            save_frames_to_folder(frames, sampling_strategy, filename)
                        except Exception as e:
                            print(f"Save frames failed: {str(e)}")

                        base64_images = frames_to_base64_images(frames, max_tokens=250000)
                        result = analyze_images_with_api(base64_images, prompt=prompt, model=model, session_id=session_id)

                        response_data = {
                            'type': 'complete',
                            'success': True,
                            'result': result,
                            'video_name': filename,
                            'model': model,
                            'input_method': 'upload',
                            'sampling_strategy': sampling_strategy,
                            'frames_extracted': len(frames),
                            'metadata': metadata
                        }

                    else:
                        result = analyze_video_with_api(video_path=final_video_path, prompt=prompt, model=model, session_id=session_id)

                        response_data = {
                            'type': 'complete',
                            'success': True,
                            'result': result,
                            'video_name': filename,
                            'model': model,
                            'input_method': 'upload'
                        }

                    if compressed_filename and os.path.exists(compressed_path):
                        response_data['compressed_video'] = compressed_filename
                        response_data['compressed_size'] = f"{os.path.getsize(compressed_path) / (1024*1024):.2f} MB"
                        response_data['original_size'] = f"{file_size_mb:.2f} MB"

                    progress_queues[session_id].put(response_data)

            except InterruptedError as e:
                # 用户主动停止分析
                print(f"\n[Session {session_id}] 分析被用户停止: {str(e)}\n")
                if compressed_path and os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass
                # 不发送错误消息，停止消息已通过 /stop 端点发送
            except Exception as e:
                print(f"\nError: {str(e)}\n")
                if compressed_path and os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass

                progress_queues[session_id].put({
                    'type': 'error',
                    'message': str(e)
                })
            finally:
                # 清理会话资源
                if session_id in stop_flags:
                    del stop_flags[session_id]

        # 启动后台线程
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()

        # 立即返回session_id
        return jsonify({
            'success': True,
            'session_id': session_id
        })

    except Exception as e:
        print(f"\n错误: {str(e)}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    """下载压缩后的视频文件"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop/<session_id>', methods=['POST'])
def stop_analysis(session_id):
    """停止指定会话的分析任务"""
    try:
        print(f"\n{'='*60}")
        print(f"收到停止请求 (Session: {session_id})")
        print(f"{'='*60}\n")

        # 设置停止标志
        stop_flags[session_id] = True

        # 向队列发送停止消息
        if session_id in progress_queues:
            progress_queues[session_id].put({
                'type': 'stopped',
                'message': '分析已停止'
            })

        return jsonify({
            'success': True,
            'message': '停止请求已发送',
            'session_id': session_id
        })

    except Exception as e:
        print(f"停止分析错误: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    has_api_key = bool(DASHSCOPE_API_KEY)
    has_ffmpeg = check_ffmpeg()
    return jsonify({
        'status': 'ok',
        'api_configured': has_api_key,
        'ffmpeg_installed': has_ffmpeg,
        'available_models': list(AVAILABLE_MODELS.keys())
    })

@app.route('/config', methods=['GET'])
def config_info():
    """获取配置信息"""
    return jsonify({
        'api_key_configured': bool(DASHSCOPE_API_KEY),
        'available_models': AVAILABLE_MODELS,
        'max_video_size_mb': MAX_VIDEO_SIZE / (1024 * 1024),
        'allowed_extensions': list(ALLOWED_EXTENSIONS)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Qwen3-VL 视频分析服务 (DashScope API)")
    print("=" * 60)
    print()

    # 检查 API Key
    if DASHSCOPE_API_KEY:
        print("✓ DashScope API Key 已配置")
        print(f"  API Key: {DASHSCOPE_API_KEY[:8]}...")
    else:
        print("✗ 警告: 未配置 DashScope API Key")
        print("  请设置环境变量:")
        print("    Windows: set DASHSCOPE_API_KEY=your_api_key")
        print("    Linux/Mac: export DASHSCOPE_API_KEY=your_api_key")
        print("  或在 app.py 中直接设置 DASHSCOPE_API_KEY 变量")
        print()
        print("  获取 API Key: https://dashscope.console.aliyun.com/apiKey")

    print()
    print(f"可用模型: {len(AVAILABLE_MODELS)} 个")
    for model_id, model_name in AVAILABLE_MODELS.items():
        print(f"  - {model_id}: {model_name}")

    # 检查 GPU 加速支持
    print()
    if check_nvenc_support():
        print("✓ NVIDIA GPU 硬件加速已启用")
        print("  视频压缩将使用 NVENC 加速 (比CPU快3-10倍)")
    else:
        print("✗ 未检测到 NVIDIA GPU 支持")
        print("  视频压缩将使用 CPU 编码 (较慢)")

    print()
    print("=" * 60)
    print("服务启动在: http://localhost:5000")
    print("=" * 60)
    print()

    # 启动 Flask 应用
    app.run(host='0.0.0.0', port=5000, debug=True)
