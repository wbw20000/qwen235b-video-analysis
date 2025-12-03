#!/usr/bin/env python
"""下载YOLOv8n模型"""

import os
import sys

def download_yolo_model():
    print("=" * 60)
    print("下载YOLOv8n模型")
    print("=" * 60)
    print()

    try:
        from ultralytics import YOLO
        print("✓ ultralytics库已加载")
    except ImportError:
        print("✗ ultralytics未安装")
        print("正在安装...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO

    print()
    print("正在下载YOLOv8n模型...")
    print("模型路径: yolov8n.pt")
    print("预计大小: ~6MB")
    print()

    try:
        # 下载预训练模型
        model = YOLO('yolov8n.pt')
        print()
        print("✓ 模型下载成功！")

        # 验证模型
        if os.path.exists('yolov8n.pt'):
            size = os.path.getsize('yolov8n.pt') / (1024*1024)
            print(f"✓ 模型文件已保存: yolov8n.pt ({size:.2f}MB)")
        else:
            print("✗ 模型文件未找到")
            return False

        return True

    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = download_yolo_model()

    print()
    print("=" * 60)
    if success:
        print("准备就绪！现在可以重启Flask应用以启用所有功能。")
    else:
        print("下载失败，请检查网络连接和磁盘空间。")
    print("=" * 60)
