"""
Qwen3-VL 视频分析 Web 应用
通过阿里云 DashScope API 调用 Qwen3-VL 模型进行视频分析
"""

import sys
import io

# 设置标准输出为 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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

# 阿里云 DashScope API 配置
# 从环境变量获取 API Key，或者直接在这里设置
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', 'sk-5175677ff9b4459aa45ce7ec28037515')

# 可用的模型列表
AVAILABLE_MODELS = {
    'qwen-vl-max': 'Qwen-VL-Max（最强视觉理解能力）',
    'qwen-vl-plus': 'Qwen-VL-Plus（推荐，性价比高）',
    'qwen3-vl-plus': 'Qwen3-VL-Plus（最新版本）'
}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        scale_width = None
        output_fps = None
        if target_bitrate < 220:
            scale_width = 480
            output_fps = 15
        elif target_bitrate < 350:
            scale_width = 640
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

def analyze_video_with_api(video_path, prompt, model='qwen-vl-plus', session_id=None):
    """
    使用阿里云 DashScope API 分析视频内容

    Args:
        video_path: 视频文件路径
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

        print(f"开始分析视频: {video_path}")
        print(f"使用模型: {model}")
        print(f"用户提问: {prompt}")

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

        # 创建 OpenAI 客户端（兼容模式）
        client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 构建请求消息
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

        # 调用 API
        print("正在调用 DashScope API...")
        send_progress(40, '正在上传到AI模型...')

        completion = client.chat.completions.create(
            model=model,
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

        # 获取用户提问和模型选择
        prompt = request.form.get('prompt', '请详细描述这个视频中发生了什么。')
        model = request.form.get('model', 'qwen-vl-plus')
        auto_compress = request.form.get('auto_compress', 'true').lower() == 'true'

        # 验证模型选择
        if model not in AVAILABLE_MODELS:
            model = 'qwen-vl-plus'

        # 保存文件
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 生成唯一的会话ID
        session_id = f"{int(time.time() * 1000)}_{os.getpid()}"

        # 创建进度队列
        progress_queues[session_id] = Queue()

        # 在后台线程中处理任务
        def process_video():
            compressed_path = None
            try:
                file_size_mb = os.path.getsize(filepath) / (1024*1024)

                print(f"\n{'='*60}")
                print(f"收到视频分析请求 (Session: {session_id})")
                print(f"视频文件: {filename}")
                print(f"保存路径: {filepath}")
                print(f"文件大小: {file_size_mb:.2f} MB")
                print(f"自动压缩: {auto_compress}")
                print(f"{'='*60}\n")

                # 检查是否需要压缩
                final_video_path = filepath
                compressed_filename = None

                if auto_compress and file_size_mb > 7.0:
                    print("视频文件较大，开始自动压缩...")

                    # 检查 FFmpeg
                    if not check_ffmpeg():
                        progress_queues[session_id].put({
                            'type': 'error',
                            'message': '未安装 FFmpeg！请先安装 FFmpeg 以使用视频压缩功能。'
                        })
                        return

                    # 生成压缩后的文件名
                    name, ext = os.path.splitext(filename)
                    compressed_filename = f"{name}_compressed{ext}"
                    compressed_path = os.path.join(app.config['UPLOAD_FOLDER'], compressed_filename)

                    # 压缩视频（带进度）
                    target_size_mb = 6.5
                    success = compress_video(filepath, compressed_path, target_size_mb=target_size_mb, session_id=session_id)

                    if success:
                        print(f"使用压缩后的视频: {compressed_filename}")
                        final_video_path = compressed_path
                        # 自适应多轮压缩，直至满足上传阈值
                        try:
                            current_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                        except Exception:
                            current_size_mb = None

                        retry = 0
                        while current_size_mb is not None and current_size_mb > TARGET_ORIGINAL_MB and retry < MAX_COMPRESS_RETRY:
                            retry += 1
                            # 动态调整目标大小：按比例收缩并留 10% 裕量，避免边界震荡
                            ratio = TARGET_ORIGINAL_MB / max(current_size_mb, 0.01)
                            new_target = max(1.0, target_size_mb * ratio * 0.9)  # 不低于 1MB，避免过低异常

                            print(f"压缩后仍为 {current_size_mb:.2f} MB，开始第 {retry} 次自适应压缩，目标 {new_target:.2f} MB")
                            progress_queues[session_id].put({
                                'type': 'compress',
                                'progress': 15,
                                'message': f'进一步压缩第{retry}次，目标 {new_target:.2f}MB'
                            })

                            # 以原始文件为输入，覆盖输出，避免多次转码累积失真
                            target_size_mb = new_target
                            success = compress_video(filepath, compressed_path, target_size_mb=target_size_mb, session_id=session_id)
                            if not success:
                                print("自适应压缩失败，保留上一版本压缩结果")
                                break

                            try:
                                current_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
                            except Exception:
                                current_size_mb = None

                        if current_size_mb is not None:
                            print(f"最终压缩文件大小: {current_size_mb:.2f} MB（阈值 {TARGET_ORIGINAL_MB:.2f} MB）")
                            if current_size_mb > TARGET_ORIGINAL_MB:
                                progress_queues[session_id].put({
                                    'type': 'compress',
                                    'progress': 95,
                                    'message': f'仍高于阈值({current_size_mb:.2f}MB>{TARGET_ORIGINAL_MB:.2f}MB)，将尝试上传，如失败请考虑截断时长或再次压缩'
                                })
                    else:
                        print("压缩失败，使用原视频")

                # 分析视频（带进度）
                result = analyze_video_with_api(final_video_path, prompt, model, session_id=session_id)

                response_data = {
                    'type': 'complete',
                    'success': True,
                    'result': result,
                    'video_name': filename,
                    'model': model
                }

                # 如果有压缩文件，返回下载链接
                if compressed_filename and os.path.exists(compressed_path):
                    response_data['compressed_video'] = compressed_filename
                    response_data['compressed_size'] = f"{os.path.getsize(compressed_path) / (1024*1024):.2f} MB"
                    response_data['original_size'] = f"{file_size_mb:.2f} MB"

                # 推送完成消息
                progress_queues[session_id].put(response_data)

            except Exception as e:
                print(f"\n错误: {str(e)}\n")
                # 清理压缩文件
                if compressed_path and os.path.exists(compressed_path):
                    try:
                        os.remove(compressed_path)
                    except:
                        pass

                # 推送错误消息
                progress_queues[session_id].put({
                    'type': 'error',
                    'message': str(e)
                })

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
