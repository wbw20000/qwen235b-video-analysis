from __future__ import annotations

import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .clip_sampler import cut_clip_ffmpeg, sample_frames_from_clip
from .config import TrafficVLMConfig
from .data_logger_and_indexer import DataLoggerAndIndexer
from .detector_and_tracker import DetectorAndTracker
from .embedding_indexer import EmbeddingIndexer
from .keyframe_extractor import extract_keyframes
from .motion_detector import MotionDetector
from .query_template_expander import expand_templates
from .temporal_clusterer import cluster_frames_to_clips
from .traffic_light_detector import TrafficLightDetector
from .video_stream_manager import VideoStreamManager, build_output_paths
from .visual_annotator import annotate_frames
from .vlm_client import VLMClient


def _default_progress(_: int, __: str):
    return


class TrafficVLMPipeline:
    """端到端管线：运动检测 -> 关键帧 -> SigLIP 检索 -> 时间聚类 -> 剪辑 -> VLM。"""

    def __init__(self, config: Optional[TrafficVLMConfig] = None, progress_cb: Optional[Callable[[int, str], None]] = None):
        self.config = config or TrafficVLMConfig()
        self.progress_cb = progress_cb or _default_progress
        self.logger = DataLoggerAndIndexer(self.config.datastore)
        self.light_detector = TrafficLightDetector()

    def _progress(self, p: int, msg: str):
        try:
            self.progress_cb(p, msg)
        except Exception:
            pass

    def run(self, video_path: str, user_query: str, camera_id: str = "camera-1") -> Dict:
        self.config.ensure_dirs()
        date_str = datetime.now().strftime("%Y%m%d")
        base_date = datetime.now()
        output_paths = build_output_paths(self.config.datastore.base_dir, camera_id, date_str)

        manager = VideoStreamManager(video_path, self.config.stream, camera_id=camera_id)
        motion_detector = MotionDetector(self.config.stream)

        self._progress(5, "启动低分辨率运动检测...")
        triggers = []
        for idx, ts, frame in manager.iterate_lowres():
            trig = motion_detector.step(frame, ts)
            triggers.extend(trig)
        self._progress(15, f"运动触发完成，触发数: {len(triggers)}")

        # 抓取高清关键帧
        keyframes = extract_keyframes(
            manager=manager,
            triggers=triggers,
            save_dir=output_paths["keyframes"],
            stream_config=self.config.stream,
            camera_id=camera_id,
            base_date=base_date,
        )
        self._progress(25, f"抓取高清关键帧完成，共 {len(keyframes)} 张")

        # 模板扩展
        templates, violation_types = expand_templates(user_query, self.config.templates)
        self._progress(30, f"扩展模板 {len(templates)} 条，违规类型 {violation_types}")

        # 向量编码与检索
        indexer = EmbeddingIndexer(self.config.embedding)
        records = [
            {
                "image_path": k["path"],
                "metadata": {
                    "camera_id": k["camera_id"],
                    "timestamp": k["timestamp"],
                    "frame_idx": k["frame_idx"],
                    "image_path": k["path"],
                    "trigger_score": k["trigger_score"],
                    "trigger_type": k["trigger_type"],
                },
            }
            for k in keyframes
        ]
        indexer.add_frame_embeddings(records)
        candidates = indexer.multi_template_search(templates)
        self._progress(45, f"帧级检索完成，候选帧 {len(candidates)}")

        # 时间聚类 -> clip
        clips = cluster_frames_to_clips(candidates, self.config.cluster)
        self._progress(55, f"时间聚类完成，候选 clip {len(clips)}")

        # 剪辑并采样
        suspect_clips: List[Dict] = []
        for clip in clips:
            out_path = os.path.join(output_paths["raw_suspect_clips"], f"{clip['clip_id']}.mp4")
            ok = cut_clip_ffmpeg(video_path, clip["start_time"], clip["end_time"], out_path)
            clip["video_path"] = out_path if ok else video_path
            suspect_clips.append(clip)
        self._progress(65, f"剪辑完成 {len(suspect_clips)} 段")

        # 本地检测/跟踪 + 采样帧
        detector = DetectorAndTracker(self.config.detector)
        vlm_client = VLMClient(self.config.vlm)

        final_results = []
        for idx, clip in enumerate(suspect_clips[: self.config.vlm.top_clips]):
            frames = sample_frames_from_clip(
                clip["video_path"],
                clip["start_time"],
                clip["end_time"],
                self.config.cluster.clip_sampling_frames,
            )
            det_result = detector.run_on_frames(frames)
            annotated = annotate_frames(
                frames,
                det_result.get("frame_results", []),
                save_dir=output_paths["annotated_frames"],
                camera_id=camera_id,
                date_str=date_str,
                roi_polygon=self.config.stream.roi_polygon,
            )
            traffic_lights = self.light_detector.detect()
            tracks_text = self._tracks_to_text(det_result.get("tracks", {}))
            traffic_light_text = self._traffic_light_to_text(traffic_lights)

            vlm_output = vlm_client.analyze(
                annotated_images=[a["path"] for a in annotated],
                intersection_info={
                    "intersection_type": "城市路口",
                    "direction_description": "北→南为正向，南→北为逆行",
                    "bike_lane_description": "右侧为非机动车道",
                },
                tracks_text=tracks_text,
                traffic_light_text=traffic_light_text,
                user_query=user_query,
            )

            # 落盘索引
            self.logger.log_event(
                {
                    "camera_id": camera_id,
                    "date": date_str,
                    "video_path": video_path,
                    "clip_id": clip["clip_id"],
                    "clip_start_time": clip["start_time"],
                    "clip_end_time": clip["end_time"],
                    "clip_score": clip["clip_score"],
                    "keyframe_paths": [k["image_path"] for k in clip["keyframes"]],
                    "motion_method": self.config.stream.motion_method,
                    "query_raw": user_query,
                    "query_templates": templates,
                    "vlm_has_violation": bool(vlm_output.get("has_violation")),
                    "vlm_violations_json": vlm_output.get("violations"),
                    "vlm_text_summary": vlm_output.get("text_summary"),
                    "traffic_light_info": traffic_lights,
                    "is_true_positive": None,
                }
            )

            final_results.append(
                {
                    "clip": clip,
                    "vlm_output": vlm_output,
                    "annotated": annotated,
                }
            )
            self._progress(70 + int(20 * (idx + 1) / max(1, len(suspect_clips))), f"VLM 分析进度 {idx+1}/{len(suspect_clips)}")

        self._progress(100, "分析完成")
        manager.release()
        return {
            "keyframes": keyframes,
            "clips": suspect_clips,
            "results": final_results,
            "templates": templates,
        }

    @staticmethod
    def _tracks_to_text(tracks: Dict) -> str:
        parts = []
        for tid, info in tracks.items():
            traj = info.get("trajectory", [])
            if traj:
                start = traj[0]
                end = traj[-1]
                parts.append(f"ID={tid}: cls={info.get('category')}, 从({start[1]:.1f},{start[2]:.1f})到({end[1]:.1f},{end[2]:.1f})")
        return "\n".join(parts)

    @staticmethod
    def _traffic_light_to_text(states: List[Dict]) -> str:
        if not states:
            return ""
        parts = [f"{s.get('timestamp', 0):.2f}s: {s.get('state')} ({s.get('confidence', 0):.2f})" for s in states]
        return "\n".join(parts)
