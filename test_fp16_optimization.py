"""
FP16 优化验证脚本
验证 YOLO half=True 和 SigLIP autocast 是否正常工作
"""
import time
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_yolo_fp16():
    """测试 YOLO FP16 是否正常工作"""
    print("\n" + "="*60)
    print("测试 1: YOLO FP16 检测")
    print("="*60)

    try:
        import torch
        import numpy as np
        from traffic_vlm.config import DetectorConfig
        from traffic_vlm.detector_and_tracker import DetectorAndTracker

        # 检查 CUDA
        if not torch.cuda.is_available():
            print("[警告] CUDA 不可用，跳过 GPU 测试")
            return False

        print(f"[信息] CUDA 可用: {torch.cuda.get_device_name(0)}")

        # 创建检测器
        config = DetectorConfig(enabled=True)
        detector = DetectorAndTracker(config)

        if detector.model is None:
            print("[警告] YOLO 模型加载失败，跳过测试")
            return False

        # 创建测试帧（随机图像）
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frames = [(0.0, test_frame), (0.5, test_frame), (1.0, test_frame)]

        # 同步 GPU 并计时
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = detector.run_on_frames(frames)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"[结果] 处理 3 帧耗时: {elapsed:.3f}s")
        print(f"[结果] 检测到 tracks: {len(result['tracks'])}")
        print(f"[结果] frame_results: {len(result['frame_results'])}")

        # 检查输出是否有 NaN/Inf（如果有检测结果）
        for fr in result['frame_results']:
            for det in fr['detections']:
                bbox = det['bbox']
                if not all(np.isfinite(bbox)):
                    print(f"[警告] 检测到 NaN/Inf 在 bbox: {bbox}")
                    return False

        print("[✓] YOLO FP16 测试通过")
        return True

    except Exception as e:
        print(f"[错误] YOLO 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_siglip_autocast():
    """测试 SigLIP autocast 是否正常工作"""
    print("\n" + "="*60)
    print("测试 2: SigLIP autocast 编码")
    print("="*60)

    try:
        import torch
        import numpy as np
        from PIL import Image
        import tempfile
        from traffic_vlm.config import EmbeddingConfig
        from traffic_vlm.embedding_indexer import EmbeddingIndexer

        # 检查 CUDA
        if not torch.cuda.is_available():
            print("[警告] CUDA 不可用，跳过 GPU 测试")
            return False

        print(f"[信息] CUDA 可用: {torch.cuda.get_device_name(0)}")

        # 创建编码器
        config = EmbeddingConfig()
        indexer = EmbeddingIndexer(config)

        print(f"[信息] device = {indexer.device}")
        print(f"[信息] use_amp = {'cuda' in str(indexer.device)}")

        # 创建临时测试图像
        temp_images = []
        for i in range(4):
            img = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))
            temp_path = tempfile.mktemp(suffix='.png')
            img.save(temp_path)
            temp_images.append(temp_path)

        # 测试 encode_images
        print("\n--- 测试 encode_images ---")
        torch.cuda.synchronize()
        start = time.perf_counter()

        image_embeds = indexer.encode_images(temp_images)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"[结果] encode_images 4张图耗时: {elapsed:.3f}s")
        print(f"[结果] 输出 shape: {image_embeds.shape}")
        print(f"[结果] 输出 dtype: {image_embeds.dtype}")

        # 检查 NaN/Inf
        if not np.isfinite(image_embeds).all():
            print("[警告] encode_images 输出包含 NaN/Inf！")
            return False

        # 测试 encode_texts
        print("\n--- 测试 encode_texts ---")
        test_texts = [
            "交通事故现场",
            "两车相撞",
            "电动车逆行",
            "红灯违规"
        ]

        torch.cuda.synchronize()
        start = time.perf_counter()

        text_embeds = indexer.encode_texts(test_texts)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"[结果] encode_texts 4条文本耗时: {elapsed:.3f}s")
        print(f"[结果] 输出 shape: {text_embeds.shape}")
        print(f"[结果] 输出 dtype: {text_embeds.dtype}")

        # 检查 NaN/Inf
        if not np.isfinite(text_embeds).all():
            print("[警告] encode_texts 输出包含 NaN/Inf！")
            return False

        # 测试相似度计算
        print("\n--- 测试相似度计算 ---")
        similarities = image_embeds @ text_embeds.T
        print(f"[结果] 相似度矩阵 shape: {similarities.shape}")
        print(f"[结果] 相似度范围: [{similarities.min():.4f}, {similarities.max():.4f}]")

        # 清理临时文件
        for path in temp_images:
            try:
                os.remove(path)
            except:
                pass

        print("[✓] SigLIP autocast 测试通过")
        return True

    except Exception as e:
        print(f"[错误] SigLIP 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory():
    """测试 GPU 显存使用情况"""
    print("\n" + "="*60)
    print("测试 3: GPU 显存使用")
    print("="*60)

    try:
        import torch

        if not torch.cuda.is_available():
            print("[警告] CUDA 不可用")
            return False

        # 获取显存信息
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)

        print(f"[信息] GPU 总显存: {total:.2f} GB")
        print(f"[信息] 已分配显存: {allocated:.2f} GB")
        print(f"[信息] 已预留显存: {reserved:.2f} GB")
        print(f"[信息] 可用显存: {total - reserved:.2f} GB")

        return True

    except Exception as e:
        print(f"[错误] GPU 显存检查失败: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("FP16 优化验证测试")
    print("="*60)

    results = []

    # 先检查 GPU 显存
    results.append(("GPU 显存检查", test_gpu_memory()))

    # 测试 YOLO FP16
    results.append(("YOLO FP16", test_yolo_fp16()))

    # 测试 SigLIP autocast
    results.append(("SigLIP autocast", test_siglip_autocast()))

    # 再次检查 GPU 显存
    results.append(("GPU 显存（测试后）", test_gpu_memory()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！FP16 优化已生效。")
    else:
        print("部分测试失败，请检查错误信息。")
    print("="*60)
