"""
GPU服务单例模块 - 避免YOLO模型重复加载

提供线程安全的GPU推理服务，所有segment共享同一个DetectorAndTracker实例。
注意：EmbeddingIndexer有状态（FAISS索引），每次pipeline运行需要新实例。
"""
from __future__ import annotations

import threading
from typing import Dict, List, Optional, Any

from .config import TrafficVLMConfig


class GPUService:
    """GPU服务单例，避免YOLO模型重复加载"""

    _instance: Optional["GPUService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: Optional[TrafficVLMConfig] = None) -> "GPUService":
        """
        获取GPU服务单例

        Args:
            config: 配置对象（首次调用时必须提供）

        Returns:
            GPUService单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if config is None:
                        raise ValueError("首次获取GPUService实例时必须提供config")
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def release(cls) -> None:
        """释放GPU资源"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._cleanup()
                cls._instance = None

    @classmethod
    def is_initialized(cls) -> bool:
        """检查GPU服务是否已初始化"""
        return cls._instance is not None

    def __init__(self, config: TrafficVLMConfig):
        """
        初始化GPU服务（私有，通过get_instance获取）

        Args:
            config: 配置对象
        """
        print("[GPUService] 初始化GPU服务（YOLO模型只加载一次）")
        self.config = config
        self._inference_lock = threading.Lock()

        # 延迟导入，避免循环依赖
        from .detector_and_tracker import DetectorAndTracker

        # 初始化YOLO检测器（单例，避免重复加载）
        if config.detector.enabled:
            print("[GPUService] 加载YOLO检测模型...")
            self.detector = DetectorAndTracker(config.detector)
        else:
            self.detector = None

        print("[GPUService] GPU服务初始化完成")

    def run_yolo(self, frames: List[tuple]) -> Dict:
        """
        线程安全的YOLO推理

        Args:
            frames: 帧列表，每个元素为 (timestamp, frame_array)

        Returns:
            检测结果字典
        """
        print(f"[GPUService] run_yolo调用，detector={'已加载' if self.detector else 'None'}, frames数量={len(frames)}")
        if self.detector is None:
            print("[GPUService] ⚠️ CRITICAL: detector为None，返回空结果")
            return {"frame_results": [], "tracks": {}}

        with self._inference_lock:
            return self.detector.run_on_frames(frames)

    def run_yolo_batch(self, frames_batch: List[Any], batch_size: int = 8) -> List[Any]:
        """
        批量YOLO推理（提高GPU利用率）

        Args:
            frames_batch: 帧批次列表
            batch_size: 批处理大小

        Returns:
            批量检测结果
        """
        if self.detector is None:
            return []

        with self._inference_lock:
            results = []
            for i in range(0, len(frames_batch), batch_size):
                batch = frames_batch[i:i+batch_size]
                # YOLO11支持批量推理
                batch_results = self.detector.model.track(
                    [f[1] for f in batch],  # 提取frame数组
                    half=True,
                    persist=True
                )
                results.extend(batch_results)
            return results

    def _cleanup(self) -> None:
        """清理GPU资源"""
        print("[GPUService] 释放GPU资源...")

        if self.detector is not None:
            del self.detector
            self.detector = None

        # 清理CUDA缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[GPUService] CUDA缓存已清理")
        except ImportError:
            pass

        print("[GPUService] GPU资源释放完成")


# 便捷函数
def get_gpu_service(config: Optional[TrafficVLMConfig] = None) -> GPUService:
    """获取GPU服务单例的便捷函数"""
    return GPUService.get_instance(config)


def release_gpu_service() -> None:
    """释放GPU服务的便捷函数"""
    GPUService.release()
