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

    def run_on_frames(self, frames: List[Tuple[float, "np.ndarray"]]) -> Dict:
        if not self.config.enabled or self.model is None:
            return {"tracks": {}, "frame_results": []}

        tracks: Dict[int, Dict] = {}
        frame_results = []

        for ts, frame in frames:
            try:
                preds = self.model.predict(
                    frame,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    verbose=False,
                )
            except Exception:
                break

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

        return {"tracks": tracks, "frame_results": frame_results}
