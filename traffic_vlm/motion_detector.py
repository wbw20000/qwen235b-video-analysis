from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional

from .config import StreamConfig


class MotionDetector:
    """MOG2/帧差/固定间隔触发器。"""

    def __init__(self, config: StreamConfig):
        self.config = config
        self.bg_subtractor = (
            cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=16, detectShadows=True)
            if self.config.motion_method == "mog2"
            else None
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.motion_streak = 0
        self.last_trigger_ts = -1e9
        self.last_force_ts = -1e9

        self.roi_mask = None
        if self.config.roi_polygon:
            # 构造 ROI mask
            mask = np.zeros((self.config.lowres_size[1], self.config.lowres_size[0]), dtype=np.uint8)
            pts = np.array(self.config.roi_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.roi_mask = mask

    def _apply_roi(self, gray: np.ndarray) -> np.ndarray:
        if self.roi_mask is None:
            return gray
        return cv2.bitwise_and(gray, self.roi_mask)

    def step(self, frame_bgr: np.ndarray, timestamp: float) -> List[Dict]:
        """处理一帧，返回触发列表。"""
        triggers: List[Dict] = []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        roi_gray = self._apply_roi(gray)

        method = self.config.motion_method
        score = 0.0

        if method == "mog2" and self.bg_subtractor is not None:
            fg_mask = self.bg_subtractor.apply(roi_gray)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg_ratio = np.count_nonzero(fg_mask) / fg_mask.size
            score = fg_ratio
            if fg_ratio >= self.config.motion_min_fg_ratio:
                self.motion_streak += 1
            else:
                self.motion_streak = 0

            if self.motion_streak >= self.config.motion_debounce_frames:
                triggers.append(
                    {
                        "timestamp": timestamp,
                        "score": float(fg_ratio),
                        "type": "motion_mog2",
                    }
                )
                self.motion_streak = 0
                self.last_trigger_ts = timestamp

        elif method == "frame_diff":
            if self.prev_gray is not None:
                diff = cv2.absdiff(roi_gray, self.prev_gray)
                score = float(np.mean(diff))
                if score >= self.config.motion_min_score:
                    self.motion_streak += 1
                else:
                    self.motion_streak = 0

                if self.motion_streak >= self.config.motion_debounce_frames:
                    triggers.append(
                        {
                            "timestamp": timestamp,
                            "score": score,
                            "type": "motion_frame_diff",
                        }
                    )
                    self.motion_streak = 0
                    self.last_trigger_ts = timestamp
            self.prev_gray = roi_gray

        elif method == "none":
            # 不做检测，固定间隔
            pass

        # 强制采样（弱信号兜底）
        if timestamp - self.last_force_ts >= self.config.always_sample_interval_seconds:
            triggers.append(
                {
                    "timestamp": timestamp,
                    "score": max(score, 0.0),
                    "type": "always_sample",
                }
            )
            self.last_force_ts = timestamp

        return triggers
