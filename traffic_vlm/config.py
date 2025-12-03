from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os


@dataclass
class StreamConfig:
    """双路流与运动检测相关配置。"""

    lowres_size: Tuple[int, int] = (640, 360)
    lowres_fps: int = 12
    motion_method: str = "mog2"  # mog2 | frame_diff | none
    motion_min_fg_ratio: float = 0.015
    motion_min_score: float = 8.0
    motion_debounce_frames: int = 2
    always_sample_interval_seconds: float = 1.0
    min_keyframe_interval: float = 2.0
    roi_polygon: Optional[List[Tuple[int, int]]] = None


@dataclass
class EmbeddingConfig:
    """SigLIP 编码与向量检索配置。"""

    model_name: str = "google/siglip-base-patch16-384"
    device: str = "auto"
    batch_size: int = 8
    top_m_per_template: int = 80
    frame_top_n: int = 80
    candidate_clip_top_k: int = 12
    clip_embedding_frames: int = 10


@dataclass
class ClusterConfig:
    """时间聚类与片段裁剪配置。"""

    merge_window_seconds: float = 5.0
    pre_padding: float = 4.0
    post_padding: float = 7.0
    clip_sampling_frames: int = 8
    candidate_clip_top_k: int = 12


@dataclass
class DetectorConfig:
    """本地检测与跟踪配置（可选）。"""

    enabled: bool = True
    model_path: str = "yolov8n.pt"
    tracker: str = "bytetrack"
    confidence_threshold: float = 0.2
    iou_threshold: float = 0.45
    max_detections: int = 100


@dataclass
class VLMConfig:
    """云端 VLM 配置。"""

    model: str = "qwen3-vl-plus"
    top_clips: int = 3
    annotated_frames_per_clip: int = 6
    temperature: float = 0.4


@dataclass
class DataStoreConfig:
    """文件与索引存储配置。"""

    base_dir: str = "data"
    sqlite_path: str = "data/index.db"

    def ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)


@dataclass
class TemplateConfig:
    """查询模板配置。"""

    builtin_templates: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "bike_wrong_way": [
                "城市道路监控画面，一辆电动自行车在机动车道中逆向行驶，与其他车辆方向相反",
                "十字路口画面，一辆二轮车面朝来车方向行驶，周围车辆方向相反",
                "路口场景，非机动车占用机动车道且方向与车流相反",
                "非机动车道逆行，二轮车反向行驶",
                "电动车在机动车道逆行",
                "自行车逆行方向与车流相反",
                "摩托车逆行上主路",
                "电动车在机动车道上逆向行驶",
            ],
            "run_red_light": [
                "红灯亮起时，一辆二轮车穿越停止线继续前行",
                "路口信号灯为红色，电动车仍然直行通过",
                "十字路口红灯，车辆/二轮车越过停止线进入路口",
                "红色信号灯时，机动车或非机动车未按规定停车",
                "路口红灯亮起，车辆继续通行",
                "信号灯变红后，车辆仍然越过停止线",
            ],
            "occupy_motor_lane": [
                "城市道路，二轮车在机动车道内行驶，旁边有汽车",
                "非机动车占用机动车道，与机动车混行",
                "机动车道上出现电动车，周围有小汽车同向行驶",
                "电动车在汽车行驶车道内行驶",
                "非机动车进入机动车专用道",
                "两轮车占用汽车道行驶",
            ],
            "accident": [
                "路口画面有车辆碰撞或行人跌倒，出现摔倒动作",
                "监控视频中两车发生碰撞，车辆或驾驶人倒地",
                "非机动车与机动车发生接触后摔倒",
                "道路上发生交通事故，车辆受损",
                "路口有车辆碰撞事件",
                "行人或骑车人摔倒在地上",
            ],
        }
    )


@dataclass
class TrafficVLMConfig:
    """整体配置聚合。"""

    stream: StreamConfig = field(default_factory=StreamConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    datastore: DataStoreConfig = field(default_factory=DataStoreConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)

    def ensure_dirs(self):
        self.datastore.ensure_dirs()
        # 预建主数据目录
        os.makedirs(self.datastore.base_dir, exist_ok=True)
