from __future__ import annotations

import os
import numpy as np
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .clip_sampler import cut_clip_ffmpeg, sample_frames_from_clip
from .config import TrafficVLMConfig
from .data_logger_and_indexer import DataLoggerAndIndexer
from .detector_and_tracker import DetectorAndTracker
from .embedding_indexer import EmbeddingIndexer
from .keyframe_extractor import extract_keyframes
from .motion_detector import MotionDetector
from .query_template_expander import expand_templates, expand_accident_templates
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
        self.accident_types = {
            "vehicle_to_vehicle_accident",
            "vehicle_to_bike_accident",
            "vehicle_to_pedestrian_accident",
            "multi_vehicle_accident",
            "hit_and_run",
        }

    def _progress(self, p: int, msg: str):
        try:
            self.progress_cb(p, msg)
        except Exception:
            pass

    def _is_accident_query(self, violation_types: List[str]) -> bool:
        return any(v in self.accident_types for v in violation_types)

    def run(self, video_path: str, user_query: str, camera_id: str = "camera-1", mode: str = "violation") -> Dict:
        """
        运行完整的视频分析管线。

        Args:
            video_path: 视频文件路径
            user_query: 用户查询意图
            camera_id: 摄像头ID
            mode: 分析模式
                  - "violation": 违法检测模式（保守策略）
                  - "accident": 事故检索模式（积极策略）
        """
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

        # 模板扩展 - 根据模式选择不同策略
        if mode == "accident":
            # 事故检索模式：使用专用的事故模板扩展
            templates, violation_types = expand_accident_templates(user_query, self.config.templates)
            accident_query = True  # 强制启用事故检测逻辑
            self._progress(30, f"[事故检索模式] 扩展模板 {len(templates)} 条，事故类型 {violation_types}")
        else:
            # 违法检测模式：使用原有逻辑
            templates, violation_types = expand_templates(user_query, self.config.templates)
            accident_query = self._is_accident_query(violation_types)
            self._progress(30, f"扩展模板 {len(templates)} 条，违规类型 {violation_types}")
        accident_templates: List[str] = []
        if accident_query:
            for v in violation_types:
                if v in self.accident_types:
                    accident_templates.extend(self.config.templates.builtin_templates.get(v, []))
            accident_templates = list(dict.fromkeys(accident_templates))

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

        # 事故模板加权召回：额外跑一轮事故模板，合并去重
        if accident_query and accident_templates:
            accident_candidates_raw = indexer.multi_template_search(accident_templates)

            def _mark_accident_candidate(c: Dict) -> Dict:
                meta = dict(c.get("metadata", {}))
                meta["accident_template_hit"] = True
                meta["accident_template_score"] = c.get("similarity_score", 0.0)
                new_c = dict(c)
                new_c["metadata"] = meta
                return new_c

            accident_candidates = [_mark_accident_candidate(c) for c in accident_candidates_raw]
            merged: Dict[str, Dict] = {}
            for item in candidates:
                path = item.get("metadata", {}).get("image_path") or item.get("frame_id")
                if path is not None:
                    merged[str(path)] = item
            for item in accident_candidates:
                path = item.get("metadata", {}).get("image_path") or item.get("frame_id")
                if path is None:
                    continue
                key = str(path)
                if key in merged:
                    # 保留相似度更高的，同时带上事故标记
                    if item["similarity_score"] > merged[key]["similarity_score"]:
                        merged[key] = item
                    else:
                        meta = dict(merged[key].get("metadata", {}))
                        meta["accident_template_hit"] = True
                        meta["accident_template_score"] = max(
                            float(meta.get("accident_template_score", 0.0) or 0.0),
                            item.get("similarity_score", 0.0),
                        )
                        merged[key]["metadata"] = meta
                else:
                    merged[key] = item
            candidates = sorted(merged.values(), key=lambda x: x["similarity_score"], reverse=True)
            candidates = candidates[: self.config.embedding.frame_top_n]

        self._progress(45, f"帧级检索完成，候选帧 {len(candidates)}")

        # 时间聚类 -> clip
        clips = cluster_frames_to_clips(candidates, self.config.cluster, accident_mode=accident_query)
        self._progress(55, f"时间聚类完成，候选 clip {len(clips)}")

        # 剪辑并采样
        suspect_clips: List[Dict] = []
        clip_success_count = 0
        clip_fail_count = 0

        for clip in clips:
            out_path = os.path.join(output_paths["raw_suspect_clips"], f"{clip['clip_id']}.mp4")
            print(f"\n[pipeline] 正在剪辑 clip: {clip['clip_id']}")
            print(f"[pipeline]   时间段: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
            if clip.get("is_accident"):
                print(f"[pipeline]   事故簇，accident_score={clip.get('accident_score', 0):.2f}")

            ok = cut_clip_ffmpeg(video_path, clip["start_time"], clip["end_time"], out_path)

            if ok:
                clip["video_path"] = out_path
                clip["clip_source"] = "clipped"
                clip_success_count += 1
            else:
                # 剪辑失败时，记录警告并使用原始视频
                print(f"[pipeline] ⚠️ 剪辑失败，将使用原始视频进行分析")
                clip["video_path"] = video_path
                clip["clip_source"] = "original_fallback"
                clip_fail_count += 1

            # 事故簇输出长版 clip，便于人工/VLM复核
            long_ver = clip.get("long_version")
            if clip.get("is_accident") and long_ver:
                long_out_path = os.path.join(output_paths["raw_suspect_clips"], f"{clip['clip_id']}_long.mp4")
                print(f"[pipeline]   生成长版 clip: {long_ver['start_time']:.2f}s - {long_ver['end_time']:.2f}s")
                ok_long = cut_clip_ffmpeg(video_path, long_ver["start_time"], long_ver["end_time"], long_out_path)
                if ok_long:
                    clip["long_video_path"] = long_out_path
                    clip["long_clip_source"] = "clipped"
                else:
                    clip["long_video_path"] = video_path
                    clip["long_clip_source"] = "original_fallback"

            suspect_clips.append(clip)

        print(f"\n[pipeline] 剪辑统计: 成功 {clip_success_count} 段, 失败 {clip_fail_count} 段")
        self._progress(65, f"剪辑完成 {clip_success_count}/{len(suspect_clips)} 段")

        # 本地检测/跟踪 + 采样帧
        detector = DetectorAndTracker(self.config.detector)
        vlm_client = VLMClient(self.config.vlm)

        final_results = []
        skipped_clips = []  # 记录因阈值过滤跳过的clip

        for idx, clip in enumerate(suspect_clips[: self.config.vlm.top_clips]):
            # P0优化：clip_score阈值过滤
            clip_score = clip.get("clip_score", 0.0)
            threshold = self.config.vlm.clip_score_threshold

            if self.config.vlm.skip_low_score_vlm and clip_score < threshold:
                print(f"[pipeline] ⏭️ 跳过 clip {clip['clip_id']}: clip_score={clip_score:.3f} < 阈值{threshold}")
                skipped_clips.append({
                    "clip": clip,
                    "reason": f"clip_score={clip_score:.3f} < {threshold}",
                    "skipped": True
                })
                continue

            # 根据clip来源决定采样时间范围
            if clip.get("clip_source") == "clipped":
                # 剪辑成功：使用相对时间（0到duration）
                sample_start = 0.0
                sample_end = clip["end_time"] - clip["start_time"]
            else:
                # 使用原始视频：使用绝对时间
                sample_start = clip["start_time"]
                sample_end = clip["end_time"]

            # 根据模式选择采样帧数：事故模式使用更密集的采样
            if mode == "accident":
                sampling_frames = self.config.cluster.accident_clip_sampling_frames
            else:
                sampling_frames = self.config.cluster.clip_sampling_frames

            frames = sample_frames_from_clip(
                clip["video_path"],
                sample_start,
                sample_end,
                sampling_frames,
            )
            det_result = detector.run_on_frames(frames)
            annotated = annotate_frames(
                frames,
                det_result.get("frame_results", []),
                save_dir=output_paths["annotated_frames"],
                camera_id=camera_id,
                date_str=date_str,
                roi_polygon=self.config.stream.roi_polygon,
                clip_id=clip["clip_id"],  # 传入clip_id避免文件名冲突
            )
            traffic_lights = self.light_detector.detect()
            tracks_text = self._tracks_to_text(det_result.get("tracks", {}))
            traffic_light_text = self._traffic_light_to_text(traffic_lights)

            # 根据模式选择不同的 VLM 分析方法
            intersection_info = {
                "intersection_type": "城市路口",
                "direction_description": "北→南为正向，南→北为逆行",
                "bike_lane_description": "右侧为非机动车道",
            }

            # 构建 clip 信息，用于 VLM 日志记录
            clip_info = {
                "clip_id": clip["clip_id"],
                "clip_index": idx,
                "total_clips": len(suspect_clips[:self.config.vlm.top_clips]),
                "video_path": clip.get("video_path"),
                "clip_source": clip.get("clip_source"),
                "start_time": clip["start_time"],
                "end_time": clip["end_time"],
                "duration": clip["end_time"] - clip["start_time"],
                "clip_score": clip.get("clip_score"),
                "is_accident": clip.get("is_accident"),
                "accident_score": clip.get("accident_score"),
                "keyframe_count": len(clip.get("keyframes", [])),
                "keyframe_paths": [k.get("image_path") for k in clip.get("keyframes", [])],
            }

            if mode == "accident":
                # 事故检索模式：使用专用的事故分析方法
                vlm_output = vlm_client.analyze_accident(
                    annotated_images=[a["path"] for a in annotated],
                    intersection_info=intersection_info,
                    tracks_text=tracks_text,
                    traffic_light_text=traffic_light_text,
                    user_query=user_query,
                    clip_info=clip_info,
                )
            else:
                # 违法检测模式：使用原有分析方法
                vlm_output = vlm_client.analyze(
                    annotated_images=[a["path"] for a in annotated],
                    intersection_info=intersection_info,
                    tracks_text=tracks_text,
                    traffic_light_text=traffic_light_text,
                    user_query=user_query,
                    clip_info=clip_info,
                )

            # 落盘索引 - 兼容两种输出格式
            has_detection = bool(vlm_output.get("has_violation") or vlm_output.get("has_accident"))
            detections = vlm_output.get("violations") or vlm_output.get("accidents") or []

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
                    "clip_accident_score": clip.get("accident_score"),
                    "clip_is_accident": clip.get("is_accident"),
                    "long_video_path": clip.get("long_video_path"),
                    "motion_method": self.config.stream.motion_method,
                    "query_raw": user_query,
                    "query_templates": templates,
                    "analysis_mode": mode,
                    "vlm_has_violation": has_detection,
                    "vlm_violations_json": detections,
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

        # 统计VLM调用情况
        vlm_stats = {
            "total_clips": len(suspect_clips[:self.config.vlm.top_clips]),
            "analyzed_clips": len(final_results),
            "skipped_clips": len(skipped_clips),
            "threshold": self.config.vlm.clip_score_threshold if self.config.vlm.skip_low_score_vlm else None
        }
        print(f"[pipeline] VLM统计: 总clips={vlm_stats['total_clips']}, 分析={vlm_stats['analyzed_clips']}, 跳过={vlm_stats['skipped_clips']}")

        return {
            "keyframes": keyframes,
            "clips": suspect_clips,
            "results": final_results,
            "skipped_clips": skipped_clips,
            "vlm_stats": vlm_stats,
            "templates": templates,
        }

    @staticmethod
    def _merge_close_tracks(tracks: Dict, distance_threshold: float = 80.0) -> Dict:
        """合并中心距离较近的轨迹（可能是同一目标的不同ID）。"""
        if not tracks:
            return {}

        track_list = list(tracks.items())
        merged = {}
        used = set()

        for tid1, info1 in track_list:
            if tid1 in used:
                continue

            traj1 = info1.get("trajectory", [])
            if not traj1:
                continue

            cluster_ids = [tid1]
            points1 = [(x, y) for _, x, y, _, _ in traj1]
            center1 = np.mean(points1, axis=0)

            for tid2, info2 in track_list:
                if tid2 in used or tid2 == tid1:
                    continue

                traj2 = info2.get("trajectory", [])
                if not traj2:
                    continue

                points2 = [(x, y) for _, x, y, _, _ in traj2]
                center2 = np.mean(points2, axis=0)
                distance = np.linalg.norm(center1 - center2)

                if distance < distance_threshold:
                    cluster_ids.append(tid2)

            merged_track = {
                "category": info1.get("category"),
                "trajectory": [],
            }
            for tid in cluster_ids:
                merged_track["trajectory"].extend(tracks[tid]["trajectory"])
                used.add(tid)

            merged_track["trajectory"].sort(key=lambda x: x[0])
            merged[tid1] = merged_track

        return merged

    @staticmethod
    def _tracks_to_text(tracks: Dict) -> str:
        if not tracks:
            return ""

        # 轨迹去重与合并
        merged_tracks = TrafficVLMPipeline._merge_close_tracks(tracks)

        parts = []
        for tid, info in merged_tracks.items():
            traj = info.get("trajectory", [])
            merged_ids = info.get("merged_ids", [tid])
            if traj:
                start = traj[0]
                end = traj[-1]
                parts.append(f"ID={tid} (含{len(merged_ids)}个轨迹): 从({start[1]:.1f},{start[2]:.1f})到({end[1]:.1f},{end[2]:.1f})")

        return "\n".join(parts)

    @staticmethod
    def _traffic_light_to_text(states: List[Dict]) -> str:
        if not states:
            return ""
        parts = [f"{s.get('timestamp', 0):.2f}s: {s.get('state')} ({s.get('confidence', 0):.2f})" for s in states]
        return "\n".join(parts)
