#!/usr/bin/env python3
"""
图片压缩工具（支持 GPU 和 CPU 回退）
- 优先使用 PyTorch GPU 加速
- 无 PyTorch 时使用 PIL CPU 处理
"""
import os
import io
import base64
from typing import Optional

from PIL import Image

# 尝试导入 PyTorch
try:
    import torch
    import torchvision.transforms.functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print('[gpu_image_utils] PyTorch 不可用，使用 CPU 压缩')


# GPU 设备缓存
_device = None


def get_device():
    """获取 GPU 设备（如果可用）"""
    global _device
    if not TORCH_AVAILABLE:
        return None
    if _device is None:
        if torch.cuda.is_available():
            gpu_id = int(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
            _device = torch.device(f'cuda:{gpu_id}' if gpu_id < torch.cuda.device_count() else 'cuda:0')
        else:
            _device = torch.device('cpu')
    return _device


def image_to_base64_url_cpu(path: str, max_width: int = 1280, quality: int = 85) -> str:
    """CPU 版本的图片压缩（使用 PIL）"""
    img = Image.open(path)
    original_size = os.path.getsize(path)
    orig_width, orig_height = img.size

    # 转换为 RGB
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # 缩放
    if orig_width > max_width:
        ratio = max_width / orig_width
        new_width = max_width
        new_height = int(orig_height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        new_width, new_height = orig_width, orig_height

    # JPEG 压缩
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    compressed_bytes = buffer.getvalue()
    compressed_size = len(compressed_bytes)

    # 日志
    saved_pct = (1 - compressed_size / original_size) * 100
    print(f'[图像压缩-CPU] {os.path.basename(path)}: {original_size//1024}KB -> {compressed_size//1024}KB ({orig_width}x{orig_height} -> {new_width}x{new_height}, 节省{saved_pct:.1f}%)')

    return f'data:image/jpeg;base64,{base64.b64encode(compressed_bytes).decode()}'


def image_to_base64_url_gpu(path: str, max_width: int = 1280, quality: int = 85, device=None) -> str:
    """GPU 版本的图片压缩（使用 PyTorch）"""
    if not TORCH_AVAILABLE:
        return image_to_base64_url_cpu(path, max_width, quality)

    if device is None:
        device = get_device()

    img = Image.open(path)
    original_size = os.path.getsize(path)
    orig_width, orig_height = img.size

    # 转换为 RGB
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    # 检查是否需要缩放
    if orig_width <= max_width:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        compressed_bytes = buffer.getvalue()
        compressed_size = len(compressed_bytes)
        saved_pct = (1 - compressed_size / original_size) * 100
        print(f'[图像压缩-GPU] {os.path.basename(path)}: {original_size//1024}KB -> {compressed_size//1024}KB (节省{saved_pct:.1f}%)')
        return f'data:image/jpeg;base64,{base64.b64encode(compressed_bytes).decode()}'

    # GPU 缩放
    ratio = max_width / orig_width
    new_width = max_width
    new_height = int(orig_height * ratio)

    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
    resized_tensor = torch.nn.functional.interpolate(
        img_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False
    )
    resized_tensor = resized_tensor.squeeze(0).cpu()
    resized_img = TF.to_pil_image(resized_tensor)

    # JPEG 压缩
    buffer = io.BytesIO()
    resized_img.save(buffer, format='JPEG', quality=quality, optimize=True)
    compressed_bytes = buffer.getvalue()
    compressed_size = len(compressed_bytes)

    saved_pct = (1 - compressed_size / original_size) * 100
    print(f'[图像压缩-GPU] {os.path.basename(path)}: {original_size//1024}KB -> {compressed_size//1024}KB ({orig_width}x{orig_height} -> {new_width}x{new_height}, 节省{saved_pct:.1f}%)')

    return f'data:image/jpeg;base64,{base64.b64encode(compressed_bytes).decode()}'


# 自动选择最佳方法
def image_to_base64_url(path: str, max_width: int = 1280, quality: int = 85) -> str:
    """自动选择 GPU 或 CPU 压缩"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return image_to_base64_url_gpu(path, max_width, quality)
    return image_to_base64_url_cpu(path, max_width, quality)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = image_to_base64_url(sys.argv[1])
        print(f'Base64 URL length: {len(result)} chars')
