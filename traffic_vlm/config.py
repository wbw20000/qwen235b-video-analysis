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
    batch_size: int = 16  # 优化：从8改为16，加速SigLIP嵌入计算
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

    # P0优化：VLM调用阈值过滤
    clip_score_threshold: float = 0.35  # clip_score低于此值跳过VLM调用
    skip_low_score_vlm: bool = True     # 是否启用阈值过滤

    # P0优化：图像压缩减少传输
    image_max_width: int = 960          # 图像最大宽度（像素）
    image_quality: int = 70             # JPEG压缩质量（1-100）
    compress_images: bool = True        # 是否启用图像压缩


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
class TsingcloudConfig:
    """云控智行API配置"""

    app_key: str = ""          # 从环境变量 TSINGCLOUD_APP_KEY 读取
    password: str = ""         # 从环境变量 TSINGCLOUD_PASSWORD 读取
    base_url: str = "https://rc.ccg.bcavt.com:8760/infraCloud"
    request_interval: float = 1.0    # 请求间隔（秒），防止限流
    poll_interval: float = 30.0      # URL轮询间隔（秒）
    poll_timeout: float = 300.0      # URL轮询超时（秒）
    verify_ssl: bool = False         # 是否验证SSL证书

    def __post_init__(self):
        # 尝试从环境变量读取凭据
        if not self.app_key:
            self.app_key = os.environ.get("TSINGCLOUD_APP_KEY", "")
        if not self.password:
            self.password = os.environ.get("TSINGCLOUD_PASSWORD", "")


@dataclass
class HistoryProcessConfig:
    """历史视频处理配置"""

    segment_duration: int = 300          # 分片时长（秒），默认5分钟
    download_retry_count: int = 3        # 下载重试次数
    download_retry_interval: float = 30.0  # 重试间隔（秒）
    analyze_timeout: float = 300.0       # 分析超时（秒）
    max_buffer_segments: int = 2         # 最大缓冲片段数
    temp_dir: str = "temp/history"       # 临时目录
    result_dir: str = "data/history_analysis"  # 结果目录
    cleanup_on_no_event: bool = True     # 无事故/违法时删除

    def ensure_dirs(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)


@dataclass
class BatchProcessConfig:
    """批量遍历处理配置"""

    max_roads_per_batch: int = 500       # 单次最大路口数
    max_cameras_per_road: int = 10       # 单路口最大摄像头数
    camera_filter: str = "panoramic"     # 摄像头过滤: panoramic=只处理全景, all=全部
    road_retry_count: int = 1            # 路口级别重试次数
    camera_retry_count: int = 1          # 摄像头级别重试次数
    skip_on_all_camera_fail: bool = True # 所有摄像头失败时跳过路口
    concurrent_cameras: int = 1          # 同时处理的摄像头数（建议1，避免API限流）
    batch_temp_dir: str = "temp/batch"   # 批量任务临时目录
    batch_result_dir: str = "data/batch_reports"  # 批量任务结果目录
    generate_summary_report: bool = True  # 生成汇总报告

    # P0优化：广度优先遍历模式
    traversal_mode: str = "breadth_first"  # depth_first=深度优先(原逻辑), breadth_first=广度优先
    cameras_per_road_first_pass: int = 1   # 第一轮每路口处理的摄像头数
    prioritize_accident_roads: bool = True # 检出事故的路口优先深入分析
    second_pass_enabled: bool = True       # 是否启用第二轮深入分析

    # 高峰时段优先遍历（由前端传入，此处为默认值）
    peak_hours_enabled: bool = False       # 是否启用高峰时段优先（前端传入时覆盖）
    default_peak_hours: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("07:00", "09:00"),  # 早高峰
        ("17:00", "19:00"),  # 晚高峰
    ])

    # 随机化选项
    randomize_road_order: bool = True      # 随机化路口遍历顺序
    randomize_camera_selection: bool = True # 随机选择摄像头

    def ensure_dirs(self):
        os.makedirs(self.batch_temp_dir, exist_ok=True)
        os.makedirs(self.batch_result_dir, exist_ok=True)


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
    tsingcloud: TsingcloudConfig = field(default_factory=TsingcloudConfig)
    history: HistoryProcessConfig = field(default_factory=HistoryProcessConfig)
    batch: BatchProcessConfig = field(default_factory=BatchProcessConfig)

    def ensure_dirs(self):
        self.datastore.ensure_dirs()
        self.history.ensure_dirs()
        self.batch.ensure_dirs()
        # 预建主数据目录
        os.makedirs(self.datastore.base_dir, exist_ok=True)
