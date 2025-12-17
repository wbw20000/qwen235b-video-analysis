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
    accident_clip_sampling_frames: int = 16  # 事故模式：更密集采样
    candidate_clip_top_k: int = 12
    max_clip_duration: float = 30.0  # 单个clip最大时长（秒），防止clip覆盖整个视频
    accident_merge_window_seconds: float = 10.0  # 事故场景放宽时间窗口
    accident_max_clip_duration: float = 45.0  # 事故场景允许更长片段
    accident_pre_padding: float = 10.0  # 事故簇默认更长的前置缓冲（增加到10秒以捕获事故前状态）
    accident_post_padding: float = 10.0  # 事故簇默认更长的后置缓冲
    accident_long_extra_pre: float = 2.0  # 事故长版额外前置缓冲
    accident_long_extra_post: float = 5.0  # 事故长版额外后置缓冲
    accident_score_weight: float = 0.6  # clip_score 中事故信号的权重
    accident_score_threshold: float = 0.35  # 判定事故簇的下限


@dataclass
class DetectorConfig:
    """本地检测与跟踪配置（可选）。"""

    enabled: bool = True
    model_path: str = "yolo11s.pt"  # 升级到YOLO11s，遮挡和异常姿态检测更好
    tracker: str = "bytetrack"
    confidence_threshold: float = 0.2  # 配置一: 置信度阈值
    iou_threshold: float = 0.45
    max_detections: int = 100
    imgsz: int = 1280  # 配置一: 检测图像尺寸


@dataclass
class VLMConfig:
    """云端 VLM 配置。"""

    model: str = "qwen3-vl-plus"
    top_clips: int = 3
    annotated_frames_per_clip: int = 6
    accident_frames_per_clip: int = 12  # 事故模式：发送更多帧给VLM
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
            # 二轮车违法行为
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
            "run_red_light_bike": [
                "红灯亮起时，一辆二轮车穿越停止线继续前行",
                "路口信号灯为红色，电动车仍然直行通过",
                "十字路口红灯，二轮车越过停止线进入路口",
                "红色信号灯时，二轮车未按规定停车",
                "路口红灯亮起，二轮车继续通行",
                "信号灯变红后，二轮车仍然越过停止线",
            ],
            "occupy_motor_lane_bike": [
                "城市道路，二轮车在机动车道内行驶，旁边有汽车",
                "非机动车占用机动车道，与机动车混行",
                "机动车道上出现电动车，周围有小汽车同向行驶",
                "电动车在汽车行驶车道内行驶",
                "非机动车进入机动车专用道",
                "两轮车占用汽车道行驶",
            ],
            "bike_improper_turning": [
                "二轮车未按规定车道转弯",
                "电动车随意变道影响其他车辆",
                "自行车违规左转或右转",
                "二轮车在路口突然转向",
                "摩托车违规变道",
            ],
            "bike_illegal_u_turn": [
                "二轮车在禁止掉头处掉头",
                "电动车在路口违规掉头",
                "自行车在路中央掉头影响交通",
                "摩托车在主干道掉头",
            ],

            # 机动车违法行为
            "car_wrong_way": [
                "机动车在禁止逆行路段逆向行驶",
                "汽车在机动车道上逆行",
                "车辆在主干道反向行驶",
                "小车在高速或快速路上逆行",
            ],
            "run_red_light_car": [
                "红灯亮起时，机动车穿越停止线继续前行",
                "路口信号灯为红色，汽车仍然直行通过",
                "十字路口红灯，机动车越过停止线进入路口",
                "红色信号灯时，机动车未按规定停车",
                "路口红灯亮起，汽车继续通行",
                "信号灯变红后，机动车仍然越过停止线",
            ],
            "illegal_parking": [
                "机动车在禁停区域违法停车",
                "车辆在禁止停车路段长时间停放",
                "路口附近违法停车影响交通",
                "机动车在黄色网格线内停车",
                "汽车在消防通道停车",
                "车辆在公交站台停车",
            ],
            "illegal_u_turn": [
                "机动车在禁止掉头处掉头",
                "汽车在路口违规掉头",
                "车辆在主干道掉头影响交通",
                "小车在隧道或桥梁处掉头",
            ],
            "speeding": [
                "机动车超速行驶",
                "汽车超过限速标志显示的速度",
                "车辆在限速路段超速",
                "小车在居民区超速行驶",
            ],
            "illegal_overtaking": [
                "机动车在禁止超车区域超车",
                "车辆在双黄线处超车",
                "汽车在弯道超车",
                "小车在视线不良处超车",
            ],
            "improper_lane_change": [
                "机动车未打转向灯变道",
                "汽车突然变道影响其他车辆",
                "车辆违规连续变道",
                "机动车在实线处变道",
            ],

            # 交通事故
            "vehicle_to_vehicle_accident": [
                "路口画面两辆机动车发生碰撞",
                "监控视频中汽车之间发生碰撞",
                "机动车追尾前车",
                "两车相撞，车辆受损",
                "十字路口车辆侧面碰撞",
            ],
            "vehicle_to_bike_accident": [
                "路口画面机动车与二轮车发生碰撞",
                "监控视频中汽车与电动车发生碰撞",
                "机动车与自行车发生接触",
                "汽车与摩托车碰撞后骑车人摔倒",
                "路口机动车撞到电动车",
            ],
            "vehicle_to_pedestrian_accident": [
                "路口画面机动车与行人发生碰撞",
                "监控视频中汽车撞到行人",
                "机动车在人行横道撞到行人",
                "车辆与行人发生交通事故",
            ],
            "multi_vehicle_accident": [
                "路口画面多车连环相撞",
                "监控视频中三辆或以上车辆连环相撞",
                "多车连撞事故",
                "车辆连环追尾事故",
            ],
            "hit_and_run": [
                "发生交通事故后车辆逃离现场",
                "肇事车辆逃逸",
                "碰撞后未停车离开",
                "事故后司机驾车逃逸",
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
