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
    clip_score_max_weight: float = 0.6  # clip_score中max的权重（0~1），0.6表示60%max+40%mean


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

    # P1-2优化：三档判决逻辑
    uncertain_threshold: float = 0.25   # confidence < uncertain_threshold → NO
                                        # uncertain_threshold <= confidence < clip_score_threshold → UNCERTAIN
                                        # confidence >= clip_score_threshold → YES (if verdict=YES)

    # P0优化：图像压缩减少传输（当前已关闭，发送原始标注图片）
    image_max_width: int = 960          # 图像最大宽度（像素）
    image_quality: int = 70             # JPEG压缩质量（1-100）
    compress_images: bool = False       # 是否启用图像压缩（False=发送原始图片）

    # VLM并行调用配置
    vlm_max_concurrent: int = 3         # VLM最大并发数（3个并发请求）

    # VLM结果置信度分级（软过滤，降低误检率）
    confidence_confirmed_threshold: float = 0.7   # >= 0.7 为"确定事故"
    confidence_suspected_threshold: float = 0.4   # 0.4-0.7 为"疑似事故"
    enable_confidence_filter: bool = True         # 是否启用置信度分级（仅事故模式）


@dataclass
class TrajectoryScoreConfig:
    """轨迹碰撞评分配置（用于降低误检率）"""

    enabled: bool = True                          # 是否启用轨迹评分
    time_window_seconds: float = 3.0              # 违法模式时间窗口
    accident_time_window_seconds: float = 4.0     # 事故模式时间窗口
    min_similarity_score: float = 0.35            # 只处理相似度>=0.35的候选

    # collision_score 参数
    collision_iou_threshold: float = 0.1          # IOU超过此值开始计分
    collision_distance_threshold: float = 100     # 中心距离阈值（像素）

    # deceleration_score 参数
    deceleration_threshold: float = 0.3           # 速度变化阈值（相对值）
    min_track_length: int = 3                     # 最小轨迹长度


@dataclass
class CoverageConfig:
    """事故过程完整性评分配置

    用于评估clip是否覆盖【事故发生前→碰撞瞬间→事故后】的完整因果链。
    包含完整过程的clip会获得更高的coverage_score，从而在排序中提升。
    """

    enabled: bool = True                          # 是否启用coverage评分

    # 目标覆盖窗口（秒）
    pre_roll: float = 8.0                         # 碰撞前需要覆盖的时长
    post_roll: float = 12.0                       # 碰撞后需要覆盖的时长

    # 最终排序分权重
    # final_score = base_score + lambda_coverage * coverage_score - mu_late * late_start_penalty
    lambda_coverage: float = 0.20                 # coverage_score权重
    mu_late: float = 0.03                         # late_start_penalty惩罚系数

    # 日志配置
    log_ranking: bool = True                      # 是否输出排名日志


@dataclass
class VLMSamplingConfig:
    """VLM t0窗口高频抽帧配置"""

    enabled: bool = True                          # 是否启用t0窗口抽帧

    # t0窗口参数
    t0_window_pre: float = 2.0                    # 碰撞前窗口（秒）
    t0_window_post: float = 3.0                   # 碰撞后窗口（秒）
    fps: int = 5                                  # 窗口内抽帧fps（5-10）

    # 额外补帧
    extra_pre: float = 6.0                        # 额外事故前帧时间点
    extra_post: float = 10.0                      # 额外事故后帧时间点

    # ROI参数
    roi_mode: str = "union"                       # union | event_window
    roi_scale: float = 1.5                        # ROI扩展比例（1.3-1.8）
    roi_window_size: int = 640                    # event_window模式下的窗口大小

    # 调试输出
    debug_dump: bool = True                       # 是否输出调试帧
    debug_dir: str = "data/debug"                 # 调试输出目录


@dataclass
class VLMRetentionConfig:
    """VLM候选保留策略配置（有条件保留）"""

    enabled: bool = True                          # 是否启用保留策略

    # 保留阈值
    accident_score_threshold: float = 1.0         # accident_score >= 此值时考虑保留
    force_keep_on_uncertain: bool = True          # verdict=UNCERTAIN时保留
    force_keep_on_post_event: bool = True         # verdict=POST_EVENT_ONLY时保留

    # 有条件保留（verdict=NO时的额外条件）
    validity_threshold: float = 0.3               # t0_validity >= 此值时保留NO
    risk_peak_threshold: float = 0.25             # risk_peak >= 此值时保留NO
    roi_median_threshold: float = 80.0            # roi_median >= 此值时保留NO

    # 严格证据要求
    strict_evidence_required: bool = True         # NO高置信但无证据时降级为UNCERTAIN

    # 保留状态
    needs_review_label: str = "NEEDS_REVIEW"      # 保留候选的标签

    # NO置信度阈值：VLM=NO且置信度>=此值时，不可被覆盖，尊重VLM判断
    no_confidence_threshold: float = 0.6          # 高置信度NO的阈值


@dataclass
class FileEventLocatorConfig:
    """文件级事件定位配置"""

    # 风险时间序列采样
    risk_sampling_fps: float = 2.0                # 采样频率(Hz)

    # 峰值检测
    top_n_peaks: int = 5                          # TopN风险峰值数量
    peak_min_distance_sec: float = 3.0            # 峰值间最小间距(秒)
    peak_threshold: float = 0.15                  # 峰值阈值(0~1)

    # 子候选窗口
    pre_roll: float = 8.0                         # 事故前覆盖时长(秒)
    post_roll: float = 12.0                       # 事故后覆盖时长(秒)
    merge_gap_sec: float = 3.0                    # 子候选合并间隔(秒)

    # 候选模式
    candidate_mode: str = "file_plus_subclips"    # file_only | file_plus_subclips

    # t0_validity 验证阈值
    min_distance_threshold: float = 150.0         # 最小距离阈值(像素)
    distance_drop_threshold: float = 0.3          # 距离突降阈值(相对值)
    velocity_change_threshold: float = 0.4        # 速度变化阈值(相对值)
    iou_contact_threshold: float = 0.05           # IoU接触阈值


@dataclass
class ProgressiveVLMConfig:
    """渐进式VLM配置 v2 - 自适应帧预算与升级策略

    目标：在不引入误报（FPR不升）的前提下，提高短FN的召回。
    核心改动：VLM输入使用"无框原始帧"+"文本化检测/轨迹元数据"。

    v2改进：
    - 自适应帧预算：根据clip时长调整S1帧数
    - 事件加帧：在高信号帧附近增加采样
    - ROI证据裁剪：提取交互热点
    """

    # 功能开关
    enabled: bool = True                          # 是否启用渐进式VLM策略
    version: str = "v2"                           # v1=固定帧数, v2=自适应帧数

    # ===== v2自适应帧预算 =====
    # S1帧数根据clip时长自适应调整
    s1_frames_short: int = 12                     # ≤12s短clip的S1帧数
    s1_frames_mid: int = 10                       # 12-30s中等clip的S1帧数
    s1_frames_long: int = 8                       # >30s长clip的S1帧数

    # clip时长分界点（秒）
    clip_duration_short: float = 12.0             # 短clip阈值
    clip_duration_mid: float = 30.0               # 中等clip阈值

    # S2帧预算（升级后使用）- v2自适应
    s2_frames_short: int = 18                     # ≤12s短clip的S2帧数
    s2_frames_mid: int = 15                       # 12-30s中等clip的S2帧数
    s2_frames_long: int = 12                      # >30s长clip的S2帧数
    s2_frames_min: int = 12                       # S2最小帧数（兜底）
    s2_frames_max: int = 24                       # S2最大帧数（上限）

    # ===== v1兼容配置（当version=v1时使用）=====
    fast_frames_min: int = 4                      # S1快速验证最小帧数
    fast_frames_max: int = 6                      # S1快速验证最大帧数
    escalated_frames_min: int = 12                # S2升级验证最小帧数
    escalated_frames_max: int = 16                # S2升级验证最大帧数

    # ===== 事件加帧配置 =====
    event_boost_enabled: bool = True              # 是否启用事件加帧
    event_boost_frames: int = 4                   # 事件附近额外加帧数
    neighbor_sec: float = 0.8                     # 事件邻域半径（秒）
    event_signal_threshold: float = 0.6           # 触发加帧的信号阈值

    # ===== ROI证据裁剪 =====
    roi_crops_enabled: bool = False               # 是否启用ROI裁剪（暂时关闭）
    roi_crops_per_stage: int = 2                  # 每阶段ROI裁剪数
    roi_scale: float = 1.5                        # ROI扩展比例

    # ===== 升级策略配置 v2 =====
    escalate_on_verdicts: List[str] = field(default_factory=lambda: ["UNCERTAIN", "POST_EVENT_ONLY"])
    escalate_on_conflict: bool = True             # risk_score高但verdict=NO时升级
    conflict_risk_threshold: float = 0.6          # conflict规则的risk阈值（降低以更敏感）

    # v2新增升级规则
    escalate_on_low_conf_no: bool = True          # verdict=NO但置信度低时升级
    low_conf_no_threshold: float = 0.5            # 低置信度NO的阈值
    escalate_on_high_signal: bool = True          # 高信号分时升级
    high_signal_threshold: float = 0.7            # 高信号分阈值

    # VLM输入形态配置
    use_overlay_frames: bool = False              # 是否使用YOLO框叠加（默认False=无框原图）
    include_object_metadata_text: bool = True     # 是否包含结构化文本元数据

    # 关键帧选择信号权重
    motion_peak_weight: float = 0.3               # 运动峰值权重
    interaction_peak_weight: float = 0.4          # 交互峰值权重（最重要）
    trajectory_break_weight: float = 0.2          # 轨迹中断权重
    post_event_cue_weight: float = 0.1            # 事故后线索权重

    # S2升级后的verdict解析规则
    # 保守策略：优先避免FPR上升
    resolution_conservative: bool = True          # 使用保守解析规则

    def get_s1_frames(self, clip_duration: float) -> int:
        """根据clip时长返回S1帧数"""
        if self.version != "v2":
            return self.fast_frames_max

        if clip_duration <= self.clip_duration_short:
            return self.s1_frames_short
        elif clip_duration <= self.clip_duration_mid:
            return self.s1_frames_mid
        else:
            return self.s1_frames_long

    def get_s2_frames(self, clip_duration: float) -> int:
        """根据clip时长返回S2帧数（v2自适应）"""
        if self.version != "v2":
            return self.escalated_frames_max

        # v2: S2帧数根据clip时长自适应
        if clip_duration <= self.clip_duration_short:
            s2 = self.s2_frames_short
        elif clip_duration <= self.clip_duration_mid:
            s2 = self.s2_frames_mid
        else:
            s2 = self.s2_frames_long

        # 确保在min-max范围内
        return max(self.s2_frames_min, min(self.s2_frames_max, s2))


@dataclass
class Stage3Config:
    """S3阶段配置 - 困难场景增强分析

    当S2返回NO但场景属于困难类型（夜间/雨天/低能见度）时，
    使用增强prompt进行S3阶段分析，避免保守漏报。
    """

    # 功能开关
    enabled: bool = False                         # 是否启用S3阶段

    # S3 门控增帧配置 (v2.2)
    s3_gate_enabled: bool = True                  # 是否启用S3门控 (False=所有clip都进S3)
    s3_gate_on_uncertain: bool = True             # S2=UNCERTAIN时触发S3
    s3_gate_on_high_signal_no: bool = True        # S2=NO但高信号时触发S3
    s3_high_signal_threshold: float = 0.65        # 高信号阈值 (final_score >= 此值)
    s3_gate_on_weather: bool = True               # 困难场景(weather prompt触发)时触发S3

    # S3 Prompt注入配置
    prompt_injection_enabled: bool = True         # 是否启用场景自适应prompt

    # 场景检测关键词（从VLM text_summary中检测）
    night_keywords: List[str] = field(default_factory=lambda: [
        "夜间", "夜晚", "夜色", "黑暗", "灯光", "车灯", "路灯", "光线不足", "low light"
    ])
    rain_keywords: List[str] = field(default_factory=lambda: [
        "雨天", "雨水", "下雨", "湿滑", "雨夜", "雨中", "rain", "wet"
    ])
    snow_keywords: List[str] = field(default_factory=lambda: [
        "雪天", "下雪", "积雪", "雪地", "冰雪", "snow"
    ])
    fog_keywords: List[str] = field(default_factory=lambda: [
        "雾天", "大雾", "浓雾", "能见度低", "fog", "visibility"
    ])

    # S3触发条件
    trigger_on_s2_no: bool = True                 # S2=NO时触发（仅在困难场景下）
    trigger_on_s2_uncertain: bool = True          # S2=UNCERTAIN时也触发

    # S3帧配置
    s3_frames: int = 16                           # S3阶段帧数

    # S3场景prompt模板
    difficult_scene_prompt_prefix: str = """【重要提示 - 困难场景分析】
当前视频存在以下困难观测条件：{scene_conditions}

在这类场景中，事故迹象可能不明显。请特别注意：
1. 即使画面模糊或光线不足，仍需仔细寻找碰撞、刮擦、人员倒地等迹象
2. 轨迹突然中断、车辆异常停止、目标消失等可能是事故信号
3. 当存在不确定性时，宁可判定为UNCERTAIN也不要轻易判定为NO
4. 如果有任何事故可能性（即使无法完全确认），应判定为YES或UNCERTAIN

请基于以上指导重新分析：
"""

    # ROI裁剪配置（S3阶段可选）
    roi_crop_enabled: bool = False                # 是否启用ROI裁剪（append模式）
    roi_scale: float = 1.5                        # ROI扩展比例
    roi_max_crops: int = 4                        # 最大ROI裁剪数

    # S3结果置信度调整
    boost_uncertain_to_yes: bool = False          # 是否将S3的UNCERTAIN提升为YES
    uncertain_boost_threshold: float = 0.7        # UNCERTAIN confidence >= 此值时提升


@dataclass
class RankScoreConfig:
    """排序评分配置（单一rank_score）

    语义对齐：
    - full_process_score: 仅当verdict=YES时才可能>0，表示完整覆盖事故过程
    - post_event_score: 仅当verdict=POST_EVENT_ONLY时=1，表示仅捕捉事故后果
    - rank_score: 唯一排序主分
    """

    # rank_score公式权重
    # rank_score = final_score + full_process_bonus - post_event_penalty + confirm_bonus
    full_process_bonus_weight: float = 0.20       # B: full_process_bonus = B * full_process_score
    post_event_penalty_weight: float = 0.20       # P: post_event_penalty = P * post_event_score
    confirm_bonus_weight: float = 0.15            # 确认事故加分权重

    # full_process_score 计算
    # full_process_score = pre_score * impact_score * post_score (仅当verdict=YES时)
    pre_roll: float = 8.0                         # 碰撞前目标覆盖时长
    post_roll: float = 12.0                       # 碰撞后目标覆盖时长
    impact_window: float = 2.0                    # 碰撞窗口半宽（秒）

    # 前置分 pre_score = min(1.0, t0 / pre_roll)
    # 冲击分 impact_score = t0_validity
    # 后置分 post_score = min(1.0, (duration - t0) / post_roll)

    # 确认加分条件
    confirm_verdict_bonus: float = 0.10           # verdict=YES 加分
    uncertain_verdict_bonus: float = 0.05         # verdict=UNCERTAIN 加分


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
    """云控智行API配置

    双账号模式：
    - HTTP轮询账号 (app_key/password): 用于 /v2x/platform/device/road/info 接口
    - RTSP账号 (rtsp_app_key/rtsp_password): 用于 /monitorModel/singleCameraQuery2 接口

    下载策略：先尝试RTSP（更快），失败后回退到HTTP轮询
    """

    # HTTP轮询账号（用于 videoType=2,3 接口）
    app_key: str = "wangbowen"
    password: str = "YwKSBcgWUI6"

    # RTSP账号（用于 singleCameraQuery2 接口）
    rtsp_app_key: str = "wangweiran"
    rtsp_password: str = "zJ952v9eFOi"

    # 设备映射文件路径（RTSP需要将roadId转换为deviceId）
    # 默认为空，会在__post_init__中设置相对路径
    device_mapping_file: str = ""

    base_url: str = "https://rc.ccg.bcavt.com:8760/infraCloud"
    request_interval: float = 1.0    # 请求间隔（秒），防止限流
    poll_interval: float = 30.0      # URL轮询间隔（秒）
    poll_timeout: float = 300.0      # URL轮询超时（秒）
    verify_ssl: bool = False         # 是否验证SSL证书
    enable_rtsp: bool = True         # 启用RTSP下载（先RTSP后轮询）

    def __post_init__(self):
        # 尝试从环境变量读取凭据（环境变量优先）
        env_key = os.environ.get("TSINGCLOUD_APP_KEY", "")
        env_pwd = os.environ.get("TSINGCLOUD_PASSWORD", "")
        if env_key:
            self.app_key = env_key
        if env_pwd:
            self.password = env_pwd
        # RTSP账号也支持环境变量
        rtsp_key = os.environ.get("TSINGCLOUD_RTSP_APP_KEY", "")
        rtsp_pwd = os.environ.get("TSINGCLOUD_RTSP_PASSWORD", "")
        if rtsp_key:
            self.rtsp_app_key = rtsp_key
        if rtsp_pwd:
            self.rtsp_password = rtsp_pwd

        # 设备映射文件：优先使用环境变量，其次使用相对路径默认值
        env_mapping = os.environ.get("TSINGCLOUD_DEVICE_MAPPING_FILE", "")
        if env_mapping:
            self.device_mapping_file = env_mapping
        elif not self.device_mapping_file:
            # 使用相对路径（项目根目录下的子目录）
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(project_root, "车网路口视频流相关资料", "deviceDJKK.csv")
            if os.path.exists(default_path):
                self.device_mapping_file = default_path


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

    # 并行下载配置
    max_concurrent_downloads: int = 5    # 并行下载数（5路）
    prefetch_segments: int = 2           # 预取片段数
    max_concurrent_per_camera: int = 2   # 同一摄像头最大并行下载数（避免API限流）

    # 视频缓存配置
    video_cache_dir: str = "cache/videos"    # 缓存目录
    min_video_size: int = 100 * 1024         # 最小有效文件大小(100KB)
    enable_video_cache: bool = True          # 启用缓存

    # RTSP下载超时配置
    rtsp_download_buffer: int = 120          # RTSP下载缓冲时间（秒）

    def get_rtsp_download_timeout(self) -> int:
        """计算RTSP下载超时 = 分片时长 + 缓冲时间"""
        return self.segment_duration + self.rtsp_download_buffer

    def ensure_dirs(self):
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        if self.enable_video_cache:
            os.makedirs(self.video_cache_dir, exist_ok=True)


@dataclass
class BatchProcessConfig:
    """批量遍历处理配置"""

    max_roads_per_batch: int = 500       # 单次最大路口数
    max_cameras_per_road: int = 10       # 单路口最大摄像头数
    camera_filter: str = "panoramic"     # 摄像头过滤: panoramic=只处理全景, all=全部
    road_retry_count: int = 1            # 路口级别重试次数
    camera_retry_count: int = 1          # 摄像头级别重试次数
    skip_on_all_camera_fail: bool = True # 所有摄像头失败时跳过路口
    concurrent_cameras: int = 2          # 同时处理的摄像头数（RTSP模式下生效，HTTP模式强制为1）
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

    # 优先路口列表：这些路口会优先遍历（在随机化之前排到最前面）
    priority_road_ids: List[str] = field(default_factory=lambda: [
        "1", "2", "5", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17",
        "24", "76", "78", "79", "113", "121", "154", "196", "207", "252", "253",
        "257", "284", "285"
    ])

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
    trajectory_score: TrajectoryScoreConfig = field(default_factory=TrajectoryScoreConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    vlm_sampling: VLMSamplingConfig = field(default_factory=VLMSamplingConfig)
    vlm_retention: VLMRetentionConfig = field(default_factory=VLMRetentionConfig)
    file_event: FileEventLocatorConfig = field(default_factory=FileEventLocatorConfig)
    rank_score: RankScoreConfig = field(default_factory=RankScoreConfig)
    progressive_vlm: ProgressiveVLMConfig = field(default_factory=ProgressiveVLMConfig)
    stage3: Stage3Config = field(default_factory=Stage3Config)

    def ensure_dirs(self):
        self.datastore.ensure_dirs()
        self.history.ensure_dirs()
        self.batch.ensure_dirs()
        # 预建主数据目录
        os.makedirs(self.datastore.base_dir, exist_ok=True)
