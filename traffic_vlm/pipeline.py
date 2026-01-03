from __future__ import annotations

import os
import re
import sys
import time
import gzip
import json
import tempfile
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

# [Windows编码修复] 安全的print函数，避免flush=True在非UTF-8控制台时触发Errno 22
def _safe_print(*args, **kwargs):
    """安全的print包装，处理Windows控制台编码问题"""
    try:
        print(*args, **kwargs)
    except OSError:
        # Windows控制台不支持某些字符时会触发OSError
        # 尝试去除flush参数后重试
        kwargs.pop('flush', None)
        try:
            # 将非ASCII字符替换为?
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
                else:
                    safe_args.append(arg)
            print(*safe_args, **kwargs)
        except Exception:
            pass  # 彻底失败时静默忽略

from .clip_sampler import cut_clip_ffmpeg, sample_frames_from_clip, parallel_cut_clips
from .config import TrafficVLMConfig
from .data_logger_and_indexer import DataLoggerAndIndexer
from .detector_and_tracker import DetectorAndTracker
from .embedding_indexer import EmbeddingIndexer
from .gpu_service import GPUService, get_gpu_service
from .keyframe_extractor import extract_keyframes
from .keyframe_selector import KeyframeSelector, build_metadata_pack
from .motion_detector import MotionDetector
from .query_template_expander import expand_templates, expand_accident_templates
from .temporal_clusterer import cluster_frames_to_clips
from .traffic_light_detector import TrafficLightDetector
from .video_stream_manager import VideoStreamManager, build_output_paths, create_video_stream_with_retry
from .visual_annotator import annotate_frames, save_raw_frames
from .vlm_client import VLMClient

# 诊断报告模块（可选，不影响主流程）
try:
    from reporting import ReportBuilder, RunContext, ClipResult, StageStats
    HAS_REPORTING = True
except ImportError:
    HAS_REPORTING = False


def _default_progress(_: int, __: str):
    return


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """
    清洗文件名，移除不安全字符

    Args:
        name: 原始文件名
        max_length: 最大长度

    Returns:
        安全的文件名
    """
    # 移除路径分隔符和特殊字符
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', name)
    # 移除连续的下划线和点
    safe = re.sub(r'[_.]{2,}', '_', safe)
    # 截断到最大长度
    if len(safe) > max_length:
        safe = safe[:max_length]
    # 移除首尾的点和下划线
    safe = safe.strip('._')
    return safe or "unnamed"


def _json_default(obj: Any) -> Any:
    """JSON序列化的default函数，处理datetime/Path等对象"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def save_video_result_json(
    result: Dict,
    output_dir: str,
    video_id: str,
) -> Optional[str]:
    """
    保存视频级完整结果到gzip压缩的JSON文件

    Args:
        result: pipeline.run()返回的完整结果字典
        output_dir: 输出目录（通常是 output_paths["video_results"]）
        video_id: 视频标识（用于生成文件名）

    Returns:
        保存成功返回绝对路径，失败返回None
    """
    try:
        # 清洗video_id生成安全文件名
        safe_name = _sanitize_filename(video_id)
        filename = f"{safe_name}.result.json.gz"
        final_path = os.path.join(output_dir, filename)
        abs_path = os.path.abspath(final_path)

        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 原子写入：先写临时文件，再rename
        fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=output_dir)
        try:
            with os.fdopen(fd, 'wb') as tmp_file:
                with gzip.GzipFile(fileobj=tmp_file, mode='wb') as gz:
                    json_str = json.dumps(result, ensure_ascii=False, indent=2, default=_json_default)
                    gz.write(json_str.encode('utf-8'))

            # 原子rename（Windows下需要先删除目标文件）
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(tmp_path, final_path)

            _safe_print(f"[SAVED] {abs_path}", flush=True)
            return abs_path

        except Exception as e:
            # 清理临时文件
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise e

    except Exception as e:
        _safe_print(f"[WARN] 保存视频结果失败: {e}", flush=True)
        return None


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

    def _compute_trajectory_scores_for_candidates(self, candidates: List[Dict], accident_query: bool):
        """
        为候选帧计算轨迹碰撞评分

        将候选帧按时间窗口分组，每组运行目标检测和跟踪，
        计算 collision_score, track_intersection_score, deceleration_score，
        并写入每个候选帧的 metadata。
        """
        import cv2
        from .trajectory_scorer import compute_trajectory_scores
        from .detector_and_tracker import DetectorAndTracker

        ts_config = self.config.trajectory_score
        time_window = (
            ts_config.accident_time_window_seconds
            if accident_query
            else ts_config.time_window_seconds
        )

        # 过滤高相似度候选
        high_score_candidates = [
            c for c in candidates
            if c.get("similarity_score", 0) >= ts_config.min_similarity_score
        ]

        if not high_score_candidates:
            print(f"[pipeline] 轨迹评分：无高相似度候选，跳过")
            return

        print(f"[pipeline] 轨迹评分：处理 {len(high_score_candidates)} 个高相似度候选")

        # 按时间窗口分组
        sorted_cands = sorted(
            high_score_candidates,
            key=lambda x: x.get("metadata", {}).get("timestamp", 0)
        )
        groups = []
        current_group = [sorted_cands[0]]

        for c in sorted_cands[1:]:
            curr_ts = c.get("metadata", {}).get("timestamp", 0)
            group_start_ts = current_group[0].get("metadata", {}).get("timestamp", 0)
            if curr_ts - group_start_ts <= time_window:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]
        groups.append(current_group)

        print(f"[pipeline] 轨迹评分：分为 {len(groups)} 个时间窗口组")

        # 初始化检测器（使用已配置的 detector config）
        detector = DetectorAndTracker(self.config.detector)

        for group_idx, group in enumerate(groups):
            # 加载帧图像
            frames = []
            for c in group:
                img_path = c.get("metadata", {}).get("image_path")
                if not img_path:
                    continue
                img = cv2.imread(img_path)
                if img is not None:
                    ts = c.get("metadata", {}).get("timestamp", 0)
                    frames.append((ts, img))

            if len(frames) < 2:
                # 帧数不足，无法有效计算轨迹
                continue

            # 运行检测和跟踪
            result = detector.run_on_frames(frames)

            # 计算评分
            scores = compute_trajectory_scores(
                result.get("tracks", {}),
                result.get("frame_results", []),
                ts_config
            )

            # 写入每个候选的 metadata
            for c in group:
                meta = c.get("metadata", {})
                meta["collision_score"] = scores["collision_score"]
                meta["track_intersection_score"] = scores["track_intersection_score"]
                meta["deceleration_score"] = scores["deceleration_score"]

            group_start_ts = group[0].get("metadata", {}).get("timestamp", 0)
            print(
                f"[pipeline] 窗口#{group_idx + 1} ({group_start_ts:.1f}s): "
                f"collision={scores['collision_score']:.2f}, "
                f"intersection={scores['track_intersection_score']:.2f}, "
                f"deceleration={scores['deceleration_score']:.2f}"
            )

    def run(
        self,
        video_path: str,
        user_query: str,
        camera_id: str = "camera-1",
        mode: str = "violation",
        generate_report: bool = False,
        report_output_dir: str = "reports",
        run_id: Optional[str] = None,
    ) -> Dict:
        """
        运行完整的视频分析管线。

        Args:
            video_path: 视频文件路径
            user_query: 用户查询意图
            camera_id: 摄像头ID
            mode: 分析模式
                  - "violation": 违法检测模式（保守策略）
                  - "accident": 事故检索模式（积极策略）
            generate_report: 是否生成诊断报告
            report_output_dir: 报告输出目录
            run_id: 运行ID（可选，不指定则自动生成）
        """
        # 初始化报告上下文
        run_start_time = datetime.now(timezone.utc)
        report_context = None
        stage_timers = {}

        if generate_report and HAS_REPORTING:
            report_context = RunContext()
            if run_id:
                report_context.run_id = run_id
            else:
                report_context.generate_run_id()
            report_context.start_time = run_start_time
            report_context.mode = mode
            report_context.camera_id = camera_id
            report_context.n_source_videos = 1
            report_context.clip_score_threshold = self.config.vlm.clip_score_threshold
            report_context.skip_low_score_vlm = self.config.vlm.skip_low_score_vlm
            report_context.top_clips = self.config.vlm.top_clips
            print(f"[pipeline] 诊断报告已启用，run_id={report_context.run_id}")

        self.config.ensure_dirs()
        date_str = datetime.now().strftime("%Y%m%d")
        base_date = datetime.now()
        output_paths = build_output_paths(self.config.datastore.base_dir, camera_id, date_str)

        # P0: 使用自动降分辨率重试机制
        manager, retry_metadata = create_video_stream_with_retry(
            video_path, self.config.stream, camera_id=camera_id
        )

        # 处理创建失败的情况
        if manager is None:
            error_msg = retry_metadata.get('error', 'Unknown error')
            _safe_print(f"[pipeline] ✗ 视频处理失败: {error_msg}", flush=True)
            return {
                "error": "视频处理失败",
                "error_detail": error_msg,
                "retry_metadata": retry_metadata,
                "keyframes": [],
                "clips": [],
                "results": [],
                "skipped_clips": [],
                "vlm_stats": {"total_clips": 0, "analyzed_clips": 0, "skipped_clips": 0},
                "templates": [],
            }

        # 记录重试信息
        if retry_metadata.get('retry'):
            _safe_print(f"[pipeline] ⚠ 使用降分辨率模式处理 (scale={retry_metadata['scale_used']})", flush=True)

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

        # 向量编码与检索（使用内存图像，避免重复磁盘I/O）
        indexer = EmbeddingIndexer(self.config.embedding)
        records = [
            {
                "image_path": k["path"],
                "image": k.get("image"),  # 内存图像（来自extract_keyframes）
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

        # 编码完成后释放内存图像（避免内存占用）
        for k in keyframes:
            if "image" in k:
                del k["image"]
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

        # 轨迹碰撞评分（仅事故模式且启用时）
        if self.config.trajectory_score.enabled and accident_query and candidates:
            self._compute_trajectory_scores_for_candidates(candidates, accident_query)
            self._progress(47, f"轨迹评分完成")

        # 时间聚类 -> clip（事故模式下启用coverage评分）
        coverage_config = self.config.coverage if accident_query else None
        clips = cluster_frames_to_clips(
            candidates,
            self.config.cluster,
            accident_mode=accident_query,
            coverage_config=coverage_config,
        )
        self._progress(55, f"时间聚类完成，候选 clip {len(clips)}")

        # 剪辑并采样 - 使用多进程并行剪辑
        print(f"\n[pipeline] 开始并行剪辑 {len(clips)} 个 clips...")

        # 并行剪辑主clips
        suspect_clips, clip_success_count, clip_fail_count = parallel_cut_clips(
            clips=clips,
            video_path=video_path,
            output_dir=output_paths["raw_suspect_clips"],
            max_workers=4
        )

        # 并行剪辑 long_version（事故clips的扩展版本）
        long_clips_to_cut = []
        for clip in suspect_clips:
            long_ver = clip.get("long_version")
            if clip.get("is_accident") and long_ver:
                long_clips_to_cut.append({
                    "clip_id": f"{clip['clip_id']}_long",
                    "start_time": long_ver["start_time"],
                    "end_time": long_ver["end_time"],
                    "parent_clip_id": clip["clip_id"],  # 记录父clip
                })

        if long_clips_to_cut:
            print(f"[pipeline] 并行剪辑 {len(long_clips_to_cut)} 个 long_version clips...")
            long_results, long_success, long_fail = parallel_cut_clips(
                clips=long_clips_to_cut,
                video_path=video_path,
                output_dir=output_paths["raw_suspect_clips"],
                max_workers=4
            )

            # 将long_version结果关联回原clip
            long_result_map = {r["clip_id"].replace("_long", ""): r for r in long_results}
            for clip in suspect_clips:
                if clip["clip_id"] in long_result_map:
                    long_r = long_result_map[clip["clip_id"]]
                    clip["long_video_path"] = long_r.get("video_path", video_path)
                    clip["long_clip_source"] = long_r.get("clip_source", "original_fallback")

            clip_success_count += long_success
            clip_fail_count += long_fail

        print(f"\n[pipeline] 剪辑统计: 成功 {clip_success_count} 段, 失败 {clip_fail_count} 段")
        self._progress(65, f"剪辑完成 {clip_success_count}/{len(suspect_clips)} 段")

        # 本地检测/跟踪 + 采样帧（使用GPU服务单例，避免重复加载模型）
        gpu_service = get_gpu_service(self.config)
        vlm_client = VLMClient(self.config.vlm)

        final_results = []
        skipped_clips = []  # 记录因阈值过滤跳过的clip

        # ============ 阶段A：预处理所有clips（串行，使用GPU） ============
        preprocessed_clips = []
        clips_to_process = suspect_clips[: self.config.vlm.top_clips]
        total_clips = len(clips_to_process)

        print(f"\n[pipeline] ========== 阶段A：预处理 {total_clips} 个clips ==========")

        for idx, clip in enumerate(clips_to_process):
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

            # [C修复] 包裹预处理在try-except中，异常降级为UNCERTAIN
            engine_error = None
            try:
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
                clip_duration = clip["end_time"] - clip["start_time"]

                if mode == "accident":
                    # 渐进式VLM v2: 根据clip时长自适应帧预算
                    progressive_config = self.config.progressive_vlm
                    if progressive_config.enabled and progressive_config.version == "v2":
                        # v2策略：短clip需要更多帧
                        v2_frames = progressive_config.get_s1_frames(clip_duration)
                        # 为了确保v2有足够帧可选，采样帧数取max(v2帧数, 配置帧数)
                        sampling_frames = max(v2_frames, self.config.cluster.accident_clip_sampling_frames)
                        # 额外安全边际：至少要有足够的帧给v2选择
                        sampling_frames = max(sampling_frames, 16)
                        print(f"[pipeline] v2自适应采样: clip_duration={clip_duration:.1f}s → sampling_frames={sampling_frames}")
                    else:
                        sampling_frames = self.config.cluster.accident_clip_sampling_frames
                else:
                    sampling_frames = self.config.cluster.clip_sampling_frames

                frames = sample_frames_from_clip(
                    clip["video_path"],
                    sample_start,
                    sample_end,
                    sampling_frames,
                )
                det_result = gpu_service.run_yolo(frames)

                # 渐进式VLM策略：保存原始帧（无YOLO叠加）
                progressive_config = self.config.progressive_vlm
                raw_frames_dir = os.path.join(output_paths["annotated_frames"], "raw")
                raw_images = []
                if progressive_config.enabled and mode == "accident":
                    # 保存原始帧用于渐进式VLM
                    raw_saved = save_raw_frames(
                        frames,
                        save_dir=raw_frames_dir,
                        camera_id=camera_id,
                        date_str=date_str,
                        clip_id=clip["clip_id"],
                    )
                    raw_images = [r["path"] for r in raw_saved]

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

                # 渐进式VLM：使用关键帧选择器选帧并构建元数据包
                metadata_text_fast = ""
                metadata_text_escalated = ""
                selected_frames_fast = []
                selected_frames_escalated = []
                max_signal_score = 0.0  # v2升级规则使用

                if progressive_config.enabled and mode == "accident":
                    keyframe_selector = KeyframeSelector(progressive_config)
                    frame_results = det_result.get("frame_results", [])
                    tracks = det_result.get("tracks", {})
                    fps = 30.0  # 默认帧率

                    # [A1修复] 从frames中提取实际时间戳（相对于clip起点）
                    # frames是[(ts, frame), ...]，ts是绝对时间，需转为clip内相对时间
                    frame_timestamps = [ts - sample_start for ts, frame in frames] if frames else None

                    # S1: FAST模式帧选择
                    selected_fast = keyframe_selector.select_frames_for_clip(
                        frame_results=frame_results,
                        tracks=tracks,
                        clip_start_time=clip["start_time"],
                        clip_duration=clip["end_time"] - clip["start_time"],
                        fps=fps,
                        mode="FAST",
                        frame_timestamps=frame_timestamps,  # [A1修复] 传入实际时间戳
                    )
                    selected_frames_fast = [
                        {"frame_idx": r.frame_idx, "timestamp": r.timestamp, "reasons": r.reason_tags,
                         "score": r.score_components.get("combined", 0)}
                        for r in selected_fast
                    ]

                    # 计算max_signal_score（用于v2升级规则）
                    max_signal_score = max((r.score_components.get("combined", 0) for r in selected_fast), default=0)

                    # S2: ESCALATED模式帧选择
                    selected_escalated = keyframe_selector.select_frames_for_clip(
                        frame_results=frame_results,
                        tracks=tracks,
                        clip_start_time=clip["start_time"],
                        clip_duration=clip["end_time"] - clip["start_time"],
                        fps=fps,
                        mode="ESCALATED",
                        frame_timestamps=frame_timestamps,  # [A1修复] 传入实际时间戳
                    )
                    selected_frames_escalated = [
                        {"frame_idx": r.frame_idx, "timestamp": r.timestamp, "reasons": r.reason_tags,
                         "score": r.score_components.get("combined", 0)}
                        for r in selected_escalated
                    ]

                    # 构建元数据包
                    metadata_text_fast = build_metadata_pack(
                        selected_fast, frame_results, tracks, progressive_config
                    )
                    metadata_text_escalated = build_metadata_pack(
                        selected_escalated, frame_results, tracks, progressive_config
                    )

                # 构建 clip 信息，用于 VLM 日志记录
                clip_info = {
                    "clip_id": clip["clip_id"],
                    "clip_index": idx,
                    "total_clips": total_clips,
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
                    # Coverage评分字段（事故模式）
                    "collision_t0": clip.get("collision_t0"),
                    "collision_t0_method": clip.get("collision_t0_method"),
                    "coverage_score": clip.get("coverage_score"),
                    "late_start_penalty": clip.get("late_start_penalty"),
                    "final_score": clip.get("final_score"),
                }

                # 收集预处理结果
                preprocessed_clips.append({
                    "clip": clip,
                    "annotated_images": [a["path"] for a in annotated],
                    "raw_images": raw_images,  # 渐进式VLM：无框原图
                    "tracks_text": tracks_text,
                    "traffic_light_text": traffic_light_text,
                    "traffic_lights": traffic_lights,
                    "clip_info": clip_info,
                    "det_result": det_result,  # 保存YOLO检测结果(bbox, track_id)
                    # 渐进式VLM数据
                    "metadata_text_fast": metadata_text_fast,
                    "metadata_text_escalated": metadata_text_escalated,
                    "selected_frames_fast": selected_frames_fast,
                    "selected_frames_escalated": selected_frames_escalated,
                    "max_signal_score": max_signal_score,  # v2升级规则使用
                    "engine_error": None,  # [C修复] 无错误
                })

                # 立即释放帧内存，避免OOM
                del frames
                del annotated

            except Exception as e:
                # [C修复] 捕获引擎错误，创建UNCERTAIN结果而非跳过
                import traceback
                engine_error = {
                    "stage": "preprocess",
                    "exc_type": type(e).__name__,
                    "exc_msg": str(e),
                    "stack_head": traceback.format_exc()[:500],
                }
                print(f"[pipeline] ⚠ clip {clip['clip_id']} 预处理异常: {type(e).__name__}: {str(e)[:100]}")

                # 构建错误时的最小clip_info
                clip_info = {
                    "clip_id": clip["clip_id"],
                    "clip_index": idx,
                    "total_clips": total_clips,
                    "start_time": clip["start_time"],
                    "end_time": clip["end_time"],
                    "duration": clip["end_time"] - clip["start_time"],
                    "clip_score": clip.get("clip_score"),
                }

                # 添加带错误标记的预处理结果，VLM阶段会处理为UNCERTAIN
                preprocessed_clips.append({
                    "clip": clip,
                    "annotated_images": [],
                    "raw_images": [],
                    "tracks_text": "",
                    "traffic_light_text": "",
                    "traffic_lights": [],
                    "clip_info": clip_info,
                    "det_result": {},
                    "metadata_text_fast": "",
                    "metadata_text_escalated": "",
                    "selected_frames_fast": [],
                    "selected_frames_escalated": [],
                    "max_signal_score": 0.0,
                    "engine_error": engine_error,  # [C修复] 记录错误信息
                })

            self._progress(65 + int(10 * (idx + 1) / max(1, total_clips)), f"预处理进度 {idx+1}/{total_clips}")

        _safe_print(f"[pipeline] 预处理完成: {len(preprocessed_clips)} 个clips待VLM分析")

        # ============ [Top-1 Fallback] 无clip通过阈值时，强制送入top1 ============
        # 通过 config.vlm.enable_top1_fallback 控制是否启用 (默认False保持向后兼容)
        enable_top1_fallback = getattr(self.config.vlm, 'enable_top1_fallback', False)
        fallback_info = {"fallback_to_top1": False, "fallback_reason": None, "fallback_clip_id": None, "fallback_clip_score": None}

        if enable_top1_fallback and len(preprocessed_clips) == 0 and len(skipped_clips) > 0 and mode == "accident":
            # 有候选clip但都因阈值被跳过 → 触发Top-1 fallback
            # 按clip_score排序，取最高的一个
            sorted_skipped = sorted(skipped_clips, key=lambda x: x["clip"].get("clip_score", 0.0), reverse=True)
            top1_clip = sorted_skipped[0]["clip"]
            top1_score = top1_clip.get("clip_score", 0.0)
            threshold = self.config.vlm.clip_score_threshold

            _safe_print(f"[pipeline] ⚡ Top-1 Fallback触发: 无clip通过阈值({threshold}), 强制送入top1 clip")
            _safe_print(f"[pipeline]   clip_id={top1_clip['clip_id']}, score={top1_score:.4f} (阈值={threshold})")

            fallback_info = {
                "fallback_to_top1": True,
                "fallback_reason": f"no_clip_pass_threshold ({len(skipped_clips)} clips all < {threshold})",
                "fallback_clip_id": top1_clip["clip_id"],
                "fallback_clip_score": {
                    "final": top1_clip.get("final_score", top1_score),
                    "base": top1_clip.get("base_score", 0.0),
                    "coverage": top1_clip.get("coverage_score", 0.0),
                }
            }

            # 对top1 clip执行预处理（复制原有预处理逻辑）
            engine_error = None
            try:
                clip = top1_clip
                if clip.get("clip_source") == "clipped":
                    sample_start = 0.0
                    sample_end = clip["end_time"] - clip["start_time"]
                else:
                    sample_start = clip["start_time"]
                    sample_end = clip["end_time"]

                clip_duration = clip["end_time"] - clip["start_time"]
                progressive_config = self.config.progressive_vlm
                if progressive_config.enabled and progressive_config.version == "v2":
                    v2_frames = progressive_config.get_s1_frames(clip_duration)
                    sampling_frames = max(v2_frames, self.config.cluster.accident_clip_sampling_frames, 16)
                else:
                    sampling_frames = self.config.cluster.accident_clip_sampling_frames

                frames = sample_frames_from_clip(clip["video_path"], sample_start, sample_end, sampling_frames)
                det_result = gpu_service.run_yolo(frames)

                raw_frames_dir = os.path.join(output_paths["annotated_frames"], "raw")
                raw_images = []
                if progressive_config.enabled:
                    raw_saved = save_raw_frames(frames, save_dir=raw_frames_dir, camera_id=camera_id,
                                               date_str=date_str, clip_id=clip["clip_id"])
                    raw_images = [r["path"] for r in raw_saved]

                annotated = annotate_frames(frames, det_result.get("frame_results", []),
                                           save_dir=output_paths["annotated_frames"], camera_id=camera_id,
                                           date_str=date_str, roi_polygon=self.config.stream.roi_polygon,
                                           clip_id=clip["clip_id"])
                traffic_lights = self.light_detector.detect()
                tracks_text = self._tracks_to_text(det_result.get("tracks", {}))
                traffic_light_text = self._traffic_light_to_text(traffic_lights)

                # 关键帧选择
                metadata_text_fast = ""
                metadata_text_escalated = ""
                selected_frames_fast = []
                selected_frames_escalated = []
                max_signal_score = 0.0

                if progressive_config.enabled:
                    keyframe_selector = KeyframeSelector(progressive_config)
                    frame_results = det_result.get("frame_results", [])
                    tracks = det_result.get("tracks", {})
                    frame_timestamps = [ts - sample_start for ts, frame in frames] if frames else None

                    selected_fast = keyframe_selector.select_frames_for_clip(
                        frame_results=frame_results, tracks=tracks, clip_start_time=clip["start_time"],
                        clip_duration=clip_duration, fps=30.0, mode="FAST", frame_timestamps=frame_timestamps)
                    selected_frames_fast = [{"frame_idx": r.frame_idx, "timestamp": r.timestamp,
                                            "reason": r.reason_tags, "score": r.score_components.get("combined", 0)} for r in selected_fast]
                    metadata_text_fast = build_metadata_pack(selected_fast, frame_results, tracks, progressive_config)
                    max_signal_score = max([r.score_components.get("combined", 0) for r in selected_fast], default=0.0)

                    selected_escalated = keyframe_selector.select_frames_for_clip(
                        frame_results=frame_results, tracks=tracks, clip_start_time=clip["start_time"],
                        clip_duration=clip_duration, fps=30.0, mode="ESCALATED", frame_timestamps=frame_timestamps)
                    selected_frames_escalated = [{"frame_idx": r.frame_idx, "timestamp": r.timestamp,
                                                 "reason": r.reason_tags, "score": r.score_components.get("combined", 0)} for r in selected_escalated]
                    metadata_text_escalated = build_metadata_pack(selected_escalated, frame_results, tracks, progressive_config)

                clip_info = {
                    "clip_id": clip["clip_id"], "start_time": clip["start_time"], "end_time": clip["end_time"],
                    "duration": clip_duration, "final_score": clip.get("final_score", top1_score),
                    "base_score": clip.get("base_score", 0.0), "coverage_score": clip.get("coverage_score", 0.0),
                    "fallback_to_top1": True,
                }

                preprocessed_clips.append({
                    "clip": clip, "annotated_images": annotated, "raw_images": raw_images,
                    "tracks_text": tracks_text, "traffic_light_text": traffic_light_text,
                    "traffic_lights": traffic_lights, "clip_info": clip_info,
                    "det_result": det_result, "metadata_text_fast": metadata_text_fast,
                    "metadata_text_escalated": metadata_text_escalated, "selected_frames_fast": selected_frames_fast,
                    "selected_frames_escalated": selected_frames_escalated, "max_signal_score": max_signal_score,
                    "engine_error": None, "fallback_to_top1": True,
                })
                _safe_print(f"[pipeline] ✓ Top-1 Fallback预处理完成: {clip['clip_id']}")

            except Exception as e:
                import traceback
                engine_error = {"stage": "fallback_preprocess", "exc_type": type(e).__name__, "exc_msg": str(e)[:200]}
                _safe_print(f"[pipeline] ✗ Top-1 Fallback预处理失败: {e}")
                traceback.print_exc()
                # 即使预处理失败也添加到队列，让VLM返回UNCERTAIN
                preprocessed_clips.append({
                    "clip": top1_clip, "annotated_images": [], "raw_images": [],
                    "tracks_text": "", "traffic_light_text": "", "traffic_lights": [],
                    "clip_info": {"clip_id": top1_clip["clip_id"], "fallback_to_top1": True},
                    "det_result": {}, "metadata_text_fast": "", "metadata_text_escalated": "",
                    "selected_frames_fast": [], "selected_frames_escalated": [], "max_signal_score": 0.0,
                    "engine_error": engine_error, "fallback_to_top1": True,
                })

        # ============ 阶段B：VLM并行调用 ============
        intersection_info = {
            "intersection_type": "城市路口",
            "direction_description": "北→南为正向，南→北为逆行",
            "bike_lane_description": "右侧为非机动车道",
        }

        if mode == "accident" and len(preprocessed_clips) > 0:
            # 事故检索模式
            progressive_config = self.config.progressive_vlm

            if progressive_config.enabled:
                # ============ 渐进式VLM策略：两阶段分析 ============
                print(f"\n[pipeline] ========== 阶段B：渐进式VLM分析 (无框原图+元数据) ==========")
                vlm_results = []

                for idx, prep in enumerate(preprocessed_clips):
                    # [C修复] 检查是否有预处理错误
                    engine_error = prep.get("engine_error")
                    if engine_error:
                        # 预处理失败，直接返回UNCERTAIN而非调用VLM
                        vlm_output = {
                            "verdict": "UNCERTAIN",
                            "confidence": 0.0,
                            "has_accident": False,
                            "accidents": [],
                            "escalated": False,
                            "engine_error": engine_error,
                            "text_summary": f"预处理失败: {engine_error.get('exc_type', 'Unknown')}: {engine_error.get('exc_msg', 'Unknown error')[:100]}",
                        }
                        print(f"[pipeline] ⚠ clip {prep['clip']['clip_id']} 因预处理错误设为UNCERTAIN")
                    else:
                        raw_images = prep.get("raw_images", [])

                        # 如果没有原始帧，回退到标注帧
                        if not raw_images:
                            print(f"[pipeline] ⚠️ clip {prep['clip']['clip_id']} 无原始帧，使用标注帧")
                            raw_images = prep["annotated_images"]

                        # 根据选帧结果获取对应的原始帧
                        fast_indices = [f["frame_idx"] for f in prep.get("selected_frames_fast", [])]
                        escalated_indices = [f["frame_idx"] for f in prep.get("selected_frames_escalated", [])]

                        # 选择对应的帧（如果索引有效）
                        raw_images_fast = [raw_images[i] for i in fast_indices if i < len(raw_images)] or raw_images[:6]
                        raw_images_escalated = [raw_images[i] for i in escalated_indices if i < len(raw_images)] or raw_images

                        # 检查是否启用S3阶段
                        stage3_config = getattr(self.config, 'stage3', None)
                        use_three_stage = stage3_config and getattr(stage3_config, 'enabled', False)

                        if use_three_stage:
                            # 三阶段渐进式VLM（支持S3困难场景增强）
                            # S3门控增帧：使用更多帧（s3_frames配置）
                            s3_frame_budget = getattr(stage3_config, 's3_frames', 16)
                            if len(raw_images) >= s3_frame_budget:
                                # 均匀采样s3_frame_budget帧
                                step = len(raw_images) / s3_frame_budget
                                s3_indices = [int(i * step) for i in range(s3_frame_budget)]
                                raw_images_s3 = [raw_images[i] for i in s3_indices if i < len(raw_images)]
                            else:
                                # 帧数不足，使用全部帧
                                raw_images_s3 = raw_images
                            metadata_text_s3 = prep.get("metadata_text_escalated", "")

                            # 将Stage3Config转为字典
                            stage3_dict = {
                                "enabled": getattr(stage3_config, 'enabled', False),
                                "prompt_injection_enabled": getattr(stage3_config, 'prompt_injection_enabled', True),
                                "night_keywords": getattr(stage3_config, 'night_keywords', []),
                                "rain_keywords": getattr(stage3_config, 'rain_keywords', []),
                                "snow_keywords": getattr(stage3_config, 'snow_keywords', []),
                                "fog_keywords": getattr(stage3_config, 'fog_keywords', []),
                                "trigger_on_s2_no": getattr(stage3_config, 'trigger_on_s2_no', True),
                                "trigger_on_s2_uncertain": getattr(stage3_config, 'trigger_on_s2_uncertain', True),
                                "difficult_scene_prompt_prefix": getattr(stage3_config, 'difficult_scene_prompt_prefix', ""),
                                "boost_uncertain_to_yes": getattr(stage3_config, 'boost_uncertain_to_yes', False),
                                "uncertain_boost_threshold": getattr(stage3_config, 'uncertain_boost_threshold', 0.7),
                            }

                            vlm_output = vlm_client.analyze_progressive_three_stage(
                                raw_images_fast=raw_images_fast,
                                raw_images_escalated=raw_images_escalated,
                                raw_images_s3=raw_images_s3,
                                metadata_text_fast=prep.get("metadata_text_fast", ""),
                                metadata_text_escalated=prep.get("metadata_text_escalated", ""),
                                metadata_text_s3=metadata_text_s3,
                                user_query=user_query,
                                clip_info=prep["clip_info"],
                                stage3_config=stage3_dict,
                                escalate_on_verdicts=progressive_config.escalate_on_verdicts,
                                escalate_on_conflict=progressive_config.escalate_on_conflict,
                                conflict_risk_threshold=progressive_config.conflict_risk_threshold,
                                resolution_conservative=progressive_config.resolution_conservative,
                                escalate_on_low_conf_no=progressive_config.escalate_on_low_conf_no,
                                low_conf_no_threshold=progressive_config.low_conf_no_threshold,
                                escalate_on_high_signal=progressive_config.escalate_on_high_signal,
                                high_signal_threshold=progressive_config.high_signal_threshold,
                                max_signal_score=prep.get("max_signal_score", 0.0),
                            )
                        else:
                            # 两阶段渐进式VLM（v2支持更多升级规则）
                            vlm_output = vlm_client.analyze_progressive_two_stage(
                                raw_images_fast=raw_images_fast,
                                raw_images_escalated=raw_images_escalated,
                                metadata_text_fast=prep.get("metadata_text_fast", ""),
                                metadata_text_escalated=prep.get("metadata_text_escalated", ""),
                                user_query=user_query,
                                clip_info=prep["clip_info"],
                                escalate_on_verdicts=progressive_config.escalate_on_verdicts,
                                escalate_on_conflict=progressive_config.escalate_on_conflict,
                                conflict_risk_threshold=progressive_config.conflict_risk_threshold,
                                resolution_conservative=progressive_config.resolution_conservative,
                                # v2新增升级规则参数
                                escalate_on_low_conf_no=progressive_config.escalate_on_low_conf_no,
                                low_conf_no_threshold=progressive_config.low_conf_no_threshold,
                                escalate_on_high_signal=progressive_config.escalate_on_high_signal,
                                high_signal_threshold=progressive_config.high_signal_threshold,
                                max_signal_score=prep.get("max_signal_score", 0.0),
                            )

                    # 记录渐进式VLM特有字段
                    vlm_output["_input_mode"] = "raw_no_overlay"
                    vlm_output["_fast_n_frames"] = len(prep.get("raw_images", [])[:6]) if not engine_error else 0
                    vlm_output["_escalated_n_frames"] = len(prep.get("raw_images", [])) if vlm_output.get("escalated") and not engine_error else 0
                    vlm_output["_selected_frames_fast"] = prep.get("selected_frames_fast", [])
                    vlm_output["_selected_frames_escalated"] = prep.get("selected_frames_escalated", [])

                    vlm_results.append(vlm_output)
                    self._progress(75 + int(15 * (idx + 1) / max(1, len(preprocessed_clips))), f"渐进式VLM {idx+1}/{len(preprocessed_clips)}")

                print(f"[pipeline] 渐进式VLM完成: {len(vlm_results)} 个结果")
                escalated_count = sum(1 for r in vlm_results if r.get("escalated"))
                print(f"[pipeline] 升级率: {escalated_count}/{len(vlm_results)} ({100*escalated_count/max(1,len(vlm_results)):.1f}%)")

            else:
                # 原有逻辑：使用批量异步并发调用（标注帧）
                max_concurrent = getattr(self.config.vlm, 'vlm_max_concurrent', 3)
                print(f"\n[pipeline] ========== 阶段B：VLM并发调用 (并发数={max_concurrent}) ==========")

                # 构建批量请求数据
                vlm_requests = [{
                    "annotated_images": p["annotated_images"],
                    "intersection_info": intersection_info,
                    "tracks_text": p["tracks_text"],
                    "traffic_light_text": p["traffic_light_text"],
                    "user_query": user_query,
                    "clip_info": p["clip_info"],
                } for p in preprocessed_clips]

                # 异步批量调用VLM
                vlm_results = vlm_client.batch_analyze_accidents_sync(
                    vlm_requests,
                    max_concurrent=max_concurrent
                )

                print(f"[pipeline] VLM批量调用完成: {len(vlm_results)} 个结果")

        elif len(preprocessed_clips) > 0:
            # 违法检测模式：保持串行调用（暂不并行化）
            print(f"\n[pipeline] ========== 阶段B：VLM串行调用 (违法检测模式) ==========")
            vlm_results = []
            for idx, prep in enumerate(preprocessed_clips):
                vlm_output = vlm_client.analyze(
                    annotated_images=prep["annotated_images"],
                    intersection_info=intersection_info,
                    tracks_text=prep["tracks_text"],
                    traffic_light_text=prep["traffic_light_text"],
                    user_query=user_query,
                    clip_info=prep["clip_info"],
                )
                vlm_results.append(vlm_output)
                self._progress(75 + int(15 * (idx + 1) / max(1, len(preprocessed_clips))), f"VLM分析 {idx+1}/{len(preprocessed_clips)}")
        else:
            vlm_results = []

        # ============ 阶段C：结果处理与落盘 ============
        print(f"\n[pipeline] ========== 阶段C：结果处理与落盘 ==========")

        for idx, (prep, vlm_output) in enumerate(zip(preprocessed_clips, vlm_results)):
            # 处理异常情况
            if isinstance(vlm_output, Exception):
                print(f"[pipeline] ⚠️ VLM调用失败 clip {prep['clip']['clip_id']}: {vlm_output}")
                vlm_output = {"has_accident": False, "accidents": [], "error": str(vlm_output)}

            clip = prep["clip"]
            traffic_lights = prep["traffic_lights"]

            # ========== 应用VLM保留策略（解耦 retain_flag 与 pred_label）==========
            if mode == "accident":
                retention_config = self.config.vlm_retention
                accident_score = clip.get("accident_score", 0.0)
                # 从vlm_output提取verdict，兼容旧格式
                verdict = vlm_output.get("verdict", "YES" if vlm_output.get("has_accident") else "NO")
                vlm_confidence = vlm_output.get("confidence", 0.0)

                # P1-2: 三档判决逻辑
                # 基于confidence将边界样本标记为UNCERTAIN
                uncertain_threshold = self.config.vlm.uncertain_threshold  # 0.25
                confirmed_threshold = self.config.vlm.clip_score_threshold  # 0.35

                if verdict == "YES":
                    if vlm_confidence < uncertain_threshold:
                        # 低置信度YES → 转为NO
                        verdict = "NO"
                        vlm_output["confidence_adjusted"] = True
                        vlm_output["original_verdict"] = "YES"
                        print(f"[pipeline] [三档判决] {clip['clip_id']} YES(conf={vlm_confidence:.2f}<{uncertain_threshold}) → NO")
                    elif vlm_confidence < confirmed_threshold:
                        # 中等置信度YES → UNCERTAIN
                        verdict = "UNCERTAIN"
                        vlm_output["confidence_adjusted"] = True
                        vlm_output["original_verdict"] = "YES"
                        print(f"[pipeline] [三档判决] {clip['clip_id']} YES(conf={vlm_confidence:.2f}) → UNCERTAIN（待复核）")
                    # else: confidence >= confirmed_threshold → 保持YES

                # 更新vlm_output中的verdict
                vlm_output["verdict"] = verdict

                # 新增：NO置信度阈值，高于此值时VLM=NO不可被覆盖
                no_conf_threshold = getattr(retention_config, "no_confidence_threshold", 0.6)

                # 初始化保留标记
                vlm_output["retain_flag"] = False
                vlm_output["retain_reason"] = ""
                vlm_output["needs_review"] = False

                # 保留策略：设置 retain_flag，但【不修改】verdict
                # 保留的含义：建议人工复核，但最终分类仍由 verdict 决定
                if retention_config.enabled and accident_score >= retention_config.accident_score_threshold:
                    if verdict == "NO":
                        # 检查VLM置信度：高置信度NO不可被覆盖
                        if vlm_confidence >= no_conf_threshold:
                            # VLM高置信度判断为NO，尊重VLM判断，不设置retain_flag
                            print(f"[pipeline] [保留策略] {clip['clip_id']} VLM=NO(conf={vlm_confidence:.2f}>={no_conf_threshold}) 高置信度，尊重VLM判断")
                        else:
                            # VLM低置信度NO，标记为需复核但【不修改verdict】
                            vlm_output["retain_flag"] = True
                            vlm_output["retain_reason"] = f"low_conf_no:accident_score={accident_score:.2f},vlm_conf={vlm_confidence:.2f}"
                            vlm_output["needs_review"] = True
                            print(f"[pipeline] [保留策略] {clip['clip_id']} VLM=NO(conf={vlm_confidence:.2f}<{no_conf_threshold}) 低置信度，标记复核(verdict保持NO)")

                # 三态verdict处理：UNCERTAIN标记需复核
                if retention_config.force_keep_on_uncertain and verdict == "UNCERTAIN":
                    vlm_output["retain_flag"] = True
                    vlm_output["retain_reason"] = "uncertain_verdict"
                    vlm_output["needs_review"] = True
                    print(f"[pipeline] [保留策略] {clip['clip_id']} VLM=UNCERTAIN 标记需人工复核")

                # 设置 has_accident 基于原始 verdict，不受 retain_flag 影响
                # 仅 YES 判断为事故，NO/UNCERTAIN 均为非事故（评测层面会区分）
                vlm_output["has_accident"] = (verdict == "YES")
            # ========== 保留策略结束 ==========

            # 落盘索引 - 现在考虑了保留策略
            has_detection = bool(vlm_output.get("has_violation") or vlm_output.get("has_accident"))
            detections = vlm_output.get("violations") or vlm_output.get("accidents") or []

            # 置信度分级（仅事故模式）
            if mode == "accident" and self.config.vlm.enable_confidence_filter and detections:
                confirmed_threshold = self.config.vlm.confidence_confirmed_threshold  # 0.7
                suspected_threshold = self.config.vlm.confidence_suspected_threshold  # 0.4

                confirmed = []
                suspected = []
                low_confidence = []

                for d in detections:
                    conf = d.get("confidence", 0)
                    if conf >= confirmed_threshold:
                        d["confidence_level"] = "confirmed"  # 确定事故
                        confirmed.append(d)
                    elif conf >= suspected_threshold:
                        d["confidence_level"] = "suspected"  # 疑似事故
                        suspected.append(d)
                    else:
                        d["confidence_level"] = "low"        # 低置信度
                        low_confidence.append(d)

                print(f"[pipeline] 置信度分级: 确定={len(confirmed)}, 疑似={len(suspected)}, 低={len(low_confidence)}")

                # 更新vlm_output - 保留所有结果但添加分级标记
                vlm_output["accidents"] = confirmed + suspected + low_confidence
                vlm_output["confirmed_accidents"] = confirmed
                vlm_output["suspected_accidents"] = suspected
                vlm_output["has_confirmed_accident"] = len(confirmed) > 0
                vlm_output["has_suspected_accident"] = len(suspected) > 0
                # 更新detections用于后续日志
                detections = vlm_output["accidents"]

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
                    "annotated_images": prep.get("annotated_images"),  # 只保留路径，节省内存
                    "det_result": prep.get("det_result"),  # YOLO检测结果(bbox, track_id)
                }
            )
            self._progress(90 + int(10 * (idx + 1) / max(1, len(preprocessed_clips))), f"落盘进度 {idx+1}/{len(preprocessed_clips)}")

        self._progress(100, "分析完成")
        manager.release()

        # P2: 性能监控统计
        processing_time = (datetime.now(timezone.utc) - run_start_time).total_seconds()
        perf_stats = {
            "processing_time_sec": processing_time,
            "video_path": video_path,
            "retry_used": retry_metadata.get('retry', False),
            "scale_used": retry_metadata.get('scale_used', 1.0),
        }

        # 尝试获取GPU内存统计
        try:
            import torch
            if torch.cuda.is_available():
                perf_stats["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                perf_stats["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
                perf_stats["gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        except:
            pass

        print(f"[pipeline] 处理耗时: {processing_time:.2f}秒")

        # 统计VLM调用情况（P1-2: 添加UNCERTAIN统计）
        uncertain_count = sum(1 for r in final_results
                             if r.get("vlm_output", {}).get("verdict") == "UNCERTAIN")
        vlm_stats = {
            "total_clips": len(suspect_clips[:self.config.vlm.top_clips]),
            "analyzed_clips": len(final_results),
            "skipped_clips": len(skipped_clips),
            "uncertain_clips": uncertain_count,  # P1-2: UNCERTAIN样本数
            "threshold": self.config.vlm.clip_score_threshold if self.config.vlm.skip_low_score_vlm else None
        }
        print(f"[pipeline] VLM统计: 总clips={vlm_stats['total_clips']}, 分析={vlm_stats['analyzed_clips']}, 跳过={vlm_stats['skipped_clips']}, UNCERTAIN={uncertain_count}")

        # 构建返回结果
        result = {
            "keyframes": keyframes,
            "clips": suspect_clips,
            "results": final_results,
            "skipped_clips": skipped_clips,
            "vlm_stats": vlm_stats,
            "templates": templates,
            "retry_metadata": retry_metadata,  # P0: 记录重试信息
            "perf_stats": perf_stats,  # P2: 性能监控统计
        }

        # ============ 生成诊断报告 ============
        if generate_report and HAS_REPORTING and report_context:
            try:
                report_context.end_time = datetime.now(timezone.utc)
                report_context.date_str = date_str

                # 更新计数
                report_context.n_clips_cut = len(suspect_clips)
                report_context.n_preprocessed = len(clips_to_process)
                report_context.n_pass_score = len(preprocessed_clips)
                report_context.n_vlm_analyzed = len(final_results)
                report_context.n_kept = sum(1 for r in final_results if r.get("vlm_output", {}).get("kept", False))
                report_context.n_topk = self.config.vlm.top_clips

                # 添加阶段统计
                report_context.add_stage(
                    "clip_sampler",
                    input_count=len(suspect_clips),
                    output_count=len([c for c in suspect_clips if c.get("clip_source") == "clipped"]),
                    details={"method": "ffmpeg_parallel"}
                )
                report_context.add_stage(
                    "preprocess_filter",
                    input_count=len(clips_to_process),
                    output_count=len(preprocessed_clips),
                    skipped_count=len(skipped_clips),
                    details={
                        "threshold": self.config.vlm.clip_score_threshold,
                        "skip_enabled": self.config.vlm.skip_low_score_vlm,
                    }
                )
                report_context.add_stage(
                    "vlm",
                    input_count=len(preprocessed_clips),
                    output_count=len(final_results),
                    details={"mode": mode}
                )

                # 构建ClipResult列表
                clip_results = []

                # 1. 所有被Top-K截断的clips标记为NOT_SELECTED
                for clip in suspect_clips[self.config.vlm.top_clips:]:
                    clip_results.append(ClipResult(
                        clip_id=clip.get("clip_id", "unknown"),
                        video_path=clip.get("video_path", ""),
                        start_time=clip.get("start_time", 0),
                        end_time=clip.get("end_time", 0),
                        duration=clip.get("end_time", 0) - clip.get("start_time", 0),
                        clip_score=clip.get("clip_score", 0),
                        accident_score=clip.get("accident_score", 0),
                        filter_status="NOT_SELECTED",
                        skip_reason=f"Top-K截断 (rank > {self.config.vlm.top_clips})",
                    ))

                # 2. 被阈值跳过的clips
                for item in skipped_clips:
                    clip = item.get("clip", {})
                    clip_results.append(ClipResult(
                        clip_id=clip.get("clip_id", "unknown"),
                        video_path=clip.get("video_path", ""),
                        start_time=clip.get("start_time", 0),
                        end_time=clip.get("end_time", 0),
                        duration=clip.get("end_time", 0) - clip.get("start_time", 0),
                        clip_score=clip.get("clip_score", 0),
                        accident_score=clip.get("accident_score", 0),
                        filter_status="SKIPPED",
                        skip_reason=item.get("reason", "clip_score_below_threshold"),
                    ))

                # 3. 通过阈值并分析的clips
                for r in final_results:
                    clip = r.get("clip", {})
                    vlm_output = r.get("vlm_output", {})
                    clip_results.append(ClipResult(
                        clip_id=clip.get("clip_id", "unknown"),
                        video_path=clip.get("video_path", ""),
                        start_time=clip.get("start_time", 0),
                        end_time=clip.get("end_time", 0),
                        duration=clip.get("end_time", 0) - clip.get("start_time", 0),
                        clip_score=clip.get("clip_score", 0),
                        accident_score=clip.get("accident_score", 0),
                        filter_status="PASSED",
                        vlm_verdict=vlm_output.get("verdict", ""),
                        vlm_confidence=vlm_output.get("confidence", 0),
                        kept=vlm_output.get("kept", False),
                        keep_reason=vlm_output.get("keep_reason", ""),
                    ))

                # 生成报告
                builder = ReportBuilder(
                    context=report_context,
                    clip_results=clip_results,
                    output_dir=report_output_dir,
                )
                report_dir = builder.build()
                result["report_dir"] = report_dir

            except Exception as e:
                print(f"[pipeline] 诊断报告生成失败（不影响主流程）: {e}")
                import traceback
                traceback.print_exc()

        # ============ 保存视频级完整结果缓存 ============
        try:
            # 从video_path提取video_id
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            result_cache_path = save_video_result_json(
                result=result,
                output_dir=output_paths["video_results"],
                video_id=video_id,
            )
            if result_cache_path:
                result["result_cache_path"] = result_cache_path
        except Exception as e:
            _safe_print(f"[WARN] 保存视频结果缓存失败（不影响主流程）: {e}", flush=True)

        return result

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
                category = info.get("category", "目标")
                # 格式优化：更醒目的ID格式，与标注帧中的"ID:X"一致
                parts.append(f"【ID:{tid}】{category} - 轨迹从({start[1]:.0f},{start[2]:.0f})到({end[1]:.0f},{end[2]:.0f})")

        return "\n".join(parts)

    @staticmethod
    def _traffic_light_to_text(states: List[Dict]) -> str:
        if not states:
            return ""
        parts = [f"{s.get('timestamp', 0):.2f}s: {s.get('state')} ({s.get('confidence', 0):.2f})" for s in states]
        return "\n".join(parts)
