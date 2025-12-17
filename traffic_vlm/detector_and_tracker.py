from __future__ import annotations

from typing import Dict, List, Tuple
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
            return {"tracks": {}, "frame_results": []}

        # 重置跟踪器，确保每个clip从干净状态开始
        self.reset_tracker()

        tracks: Dict[int, Dict] = {}
        frame_results = []

        print(f"[DetectorAndTracker] 输入帧数: {len(frames)}")

        for ts, frame in frames:
            preds = None
            try:
                # 使用 track() 而不是 predict()，启用跨帧跟踪
                # persist=True 保持跟踪器状态，使同一目标跨帧保持相同ID
                preds = self.model.track(
                    frame,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    imgsz=getattr(self.config, 'imgsz', 1280),
                    persist=True,  # 关键：保持跟踪状态
                    tracker="traffic_vlm/custom_bytetrack.yaml",  # 自定义配置：track_buffer=120
                    verbose=False,
                )
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
