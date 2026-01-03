from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import DetectorConfig

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class DetectorAndTracker:
    """
    本地检测/跟踪（可选）。默认关闭，需显式开启并提供权重。
    若未安装 ultralytics，则返回空结果。
    使用 YOLO 的 track() 方法进行跨帧跟踪，确保同一目标在不同帧保持相同ID。
    """

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.model = None
        if self.config.enabled and YOLO is not None:
            try:
                self.model = YOLO(self.config.model_path)
            except Exception:
                self.model = None

        self.next_track_id = 0

    def _new_track_id(self) -> int:
        self.next_track_id += 1
        return self.next_track_id

    def reset_tracker(self):
        """重置跟踪器状态，在处理新clip时调用"""
        if self.model is not None:
            try:
                self.model.predictor = None  # 重置predictor以清除跟踪状态
            except Exception:
                pass
        self.next_track_id = 0

    def run_on_frames(self, frames: List[Tuple[float, "np.ndarray"]]) -> Dict:
        if not self.config.enabled or self.model is None:
            print(f"[DetectorAndTracker] 检测器未启用: enabled={self.config.enabled}, model={'已加载' if self.model else 'None'}")
            return {"tracks": {}, "frame_results": []}

        # 重置跟踪器，确保每个clip从干净状态开始
        self.reset_tracker()

        tracks: Dict[int, Dict] = {}
        frame_results = []

        print(f"[DetectorAndTracker] 输入帧数: {len(frames)}")

        # 标记是否已回退到 FP32（避免每帧重复打印）
        _half_fallback_logged = False

        for ts, frame in frames:
            preds = None
            try:
                # 构建 track 参数，half 参数可能不被接收需要兼容处理
                track_kwargs = dict(
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=getattr(self.config, 'imgsz', 1280),
                    persist=True,  # 关键：保持跟踪状态
                    tracker="traffic_vlm/custom_bytetrack.yaml",  # 自定义配置：track_buffer=120
                    verbose=False,
                )
                # 尝试启用 FP16，如果 track() 不支持会忽略
                try:
                    preds = self.model.track(frame, half=True, **track_kwargs)
                except TypeError as e:
                    if "half" in str(e):
                        if not _half_fallback_logged:
                            print("[DetectorAndTracker] ⚠️ track() 不支持 half 参数，回退到 FP32")
                            _half_fallback_logged = True
                        preds = self.model.track(frame, **track_kwargs)
                    else:
                        raise
            except Exception as e:
                # 不再 break，继续处理后续帧
                print(f"[DetectorAndTracker] ⚠️ 帧 {ts:.3f}s 检测异常: {e}")

            detections = []
            if preds:
                result = preds[0]
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        cls_idx = int(box.cls[0].item()) if box.cls is not None else -1
                        score = float(box.conf[0].item()) if box.conf is not None else 0.0
                        track_id = int(box.id[0].item()) if box.id is not None else self._new_track_id()

                        detections.append(
                            {
                                "track_id": track_id,
                                "bbox": xyxy,
                                "category": cls_idx,
                                "score": score,
                                "timestamp": ts,
                            }
                        )

                        traj = tracks.setdefault(
                            track_id,
                            {
                                "category": cls_idx,
                                "trajectory": [],
                            },
                        )
                        traj["trajectory"].append((ts, *xyxy))

            frame_results.append({"timestamp": ts, "detections": detections})
            detected_ids = [d['track_id'] for d in detections]
            print(f"[DetectorAndTracker] 帧 {ts:.3f}s 检测到 {len(detections)} 个目标, IDs: {detected_ids}")

        print(f"[DetectorAndTracker] 输出 frame_results 长度: {len(frame_results)}, tracks 数量: {len(tracks)}")
        return {"tracks": tracks, "frame_results": frame_results}

    def run_on_frames_batched(
        self,
        frames: List[Tuple[float, "np.ndarray"]],
        batch_size: int = 8,
        use_tracking: bool = False
    ) -> Dict:
        """
        批量处理多帧，提高GPU利用率。

        Args:
            frames: 帧列表 [(timestamp, frame), ...]
            batch_size: 批处理大小（仅对predict模式有效）
            use_tracking: 是否使用跟踪（True时退化为逐帧处理）

        Returns:
            与 run_on_frames 相同格式的结果
        """
        if not self.config.enabled or self.model is None:
            return {"tracks": {}, "frame_results": []}

        # 如果需要跟踪，使用原有逻辑（跟踪需要按序进行）
        if use_tracking:
            return self.run_on_frames(frames)

        # 批量检测模式（不跟踪）
        print(f"[DetectorAndTracker] 批量检测模式，batch_size={batch_size}，共{len(frames)}帧")

        frame_results = []
        timestamps = [ts for ts, _ in frames]
        frame_arrays = [f for _, f in frames]

        # 批量推理
        for i in range(0, len(frame_arrays), batch_size):
            batch_frames = frame_arrays[i:i + batch_size]
            batch_ts = timestamps[i:i + batch_size]

            try:
                # 使用 predict 进行批量推理（比 track 快，但无跟踪ID）
                preds = self.model.predict(
                    batch_frames,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=getattr(self.config, 'imgsz', 1280),
                    half=True,  # FP16加速
                    verbose=False,
                )

                # 处理每帧结果
                for j, (ts, pred) in enumerate(zip(batch_ts, preds)):
                    detections = []
                    boxes = pred.boxes
                    if boxes is not None:
                        for box in boxes:
                            xyxy = box.xyxy[0].tolist()
                            cls_idx = int(box.cls[0].item()) if box.cls is not None else -1
                            score = float(box.conf[0].item()) if box.conf is not None else 0.0
                            # 无跟踪ID，使用帧内索引
                            detections.append({
                                "track_id": -1,  # 表示无跟踪
                                "bbox": xyxy,
                                "category": cls_idx,
                                "score": score,
                                "timestamp": ts,
                            })

                    frame_results.append({"timestamp": ts, "detections": detections})

            except Exception as e:
                print(f"[DetectorAndTracker] ⚠️ 批量检测异常: {e}")
                # 回退到逐帧处理
                for ts, frame in zip(batch_ts, batch_frames):
                    frame_results.append({"timestamp": ts, "detections": []})

        print(f"[DetectorAndTracker] 批量检测完成，共处理{len(frame_results)}帧")
        return {"tracks": {}, "frame_results": frame_results}

    def detect_batch(
        self,
        frames: List["np.ndarray"],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """
        纯检测模式批处理（无跟踪，最高效率）

        Args:
            frames: 帧数组列表
            batch_size: 批处理大小

        Returns:
            每帧的检测结果列表 [[{bbox, category, score}, ...], ...]
        """
        if not self.config.enabled or self.model is None:
            return [[] for _ in frames]

        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]

            try:
                preds = self.model.predict(
                    batch,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=getattr(self.config, 'imgsz', 1280),
                    half=True,
                    verbose=False,
                )

                for pred in preds:
                    frame_dets = []
                    if pred.boxes is not None:
                        for box in pred.boxes:
                            frame_dets.append({
                                "bbox": box.xyxy[0].tolist(),
                                "category": int(box.cls[0].item()),
                                "score": float(box.conf[0].item()),
                            })
                    all_detections.append(frame_dets)

            except Exception as e:
                print(f"[DetectorAndTracker] ⚠️ detect_batch异常: {e}")
                all_detections.extend([[] for _ in batch])

        return all_detections
