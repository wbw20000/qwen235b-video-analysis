"""
TrafficVLM package: 二轮车违法检索与精细识别核心组件。
各模块职责：
- config: 配置管理
- video_stream_manager: 高低分辨率双路流管理（本地/RTSP）
- motion_detector: 运动触发（MOG2/帧差/无检测）
- keyframe_extractor: 根据触发抓取高清关键帧
- query_template_expander: 违规类型模板扩展
- embedding_indexer: SigLIP 编码与向量索引
- temporal_clusterer: 时间聚类生成候选 clip
- clip_sampler: 片段采样与剪辑
- detector_and_tracker: 本地检测与跟踪（可选）
- visual_annotator: 可视化标注
- traffic_light_detector: 信号灯占位
- vlm_client: 云端 VLM 调用
- data_logger_and_indexer: 日志与索引落盘
"""

__all__ = [
    "config",
    "video_stream_manager",
    "motion_detector",
    "keyframe_extractor",
    "query_template_expander",
    "embedding_indexer",
    "temporal_clusterer",
    "clip_sampler",
    "detector_and_tracker",
    "visual_annotator",
    "traffic_light_detector",
    "vlm_client",
    "data_logger_and_indexer",
]
