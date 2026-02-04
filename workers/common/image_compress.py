#!/usr/bin/env python3
"""
Image compression using ffmpeg
No PIL required - uses ffmpeg subprocess
"""
import subprocess
import tempfile
import base64
import os


def compress_image_ffmpeg(path, max_width=1280, quality=85):
    """
    Compress image using ffmpeg
    Follows Qwen3-VL recommended max_pixels = 1280*32*32
    """
    original_size = os.path.getsize(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    
    try:
        # ffmpeg scale filter: min(max_width, iw) keeps aspect ratio
        scale_filter = "scale=min(" + str(max_width) + "\,iw):-2"
        # quality: ffmpeg -q:v range 1-31, 1 is best
        q_value = max(1, min(31, (100 - quality) // 3))
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", path,
            "-vf", scale_filter,
            "-q:v", str(q_value),
            tmp_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if result.returncode != 0:
            # ffmpeg failed, return original
            with open(path, "rb") as f:
                return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
        
        with open(tmp_path, "rb") as f:
            data = f.read()
        
        print("[Compress] %s: %dKB -> %dKB" % (os.path.basename(path), original_size//1024, len(data)//1024))
        return "data:image/jpeg;base64," + base64.b64encode(data).decode()
    except Exception:
        with open(path, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
