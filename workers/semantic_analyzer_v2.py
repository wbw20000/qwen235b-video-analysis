#!/usr/bin/env python3
"""
Semantic Analyzer V2 - 支持YAML模板和分析目录的增强版分析器

特性：
1. 使用YAML模板定义positives/hard_negatives/negatives
2. 新的评分公式：group_score = weight * (pos_sim - beta*hard_sim - gamma*neg_sim)
3. 支持ADS两阶段门控
4. 热重载支持
5. 不影响accident分析流程

使用示例：
    analyzer = SemanticAnalyzerV2(
        analysis_type="mv_violation"
    )
    analyzer.run()
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.semantic_base import (
    SemanticAnalyzerBase,
    AnalysisResult,
    FrameInfo,
    ClipInfo,
    CLIP_SCORE_THRESHOLD
)

# 延迟导入semantic模块（避免循环依赖）
_registry = None
_catalog = None


def get_template_registry():
    """获取模板注册中心（延迟加载）"""
    global _registry
    if _registry is None:
        from traffic_vlm.semantic import get_registry
        _registry = get_registry()
    return _registry


def get_analysis_catalog():
    """获取分析目录（延迟加载）"""
    global _catalog
    if _catalog is None:
        from traffic_vlm.semantic import get_catalog
        _catalog = get_catalog()
    return _catalog


log = logging.getLogger(__name__)


@dataclass
class V2ScoringResult:
    """V2评分结果"""
    final_score: float
    best_group: str
    best_score: float
    pos_sim: float
    all_group_scores: Dict[str, Any] = field(default_factory=dict)


class SemanticAnalyzerV2(SemanticAnalyzerBase):
    """
    V2语义分析器 - 使用YAML模板和新评分公式

    与基类的区别：
    - 使用TemplateRegistry加载YAML模板
    - 使用新的三元组评分公式
    - 支持从AnalysisCatalog加载prompt
    - 支持ADS门控逻辑
    """

    def __init__(
        self,
        analysis_type: str,
        consumer_group: str = None,
        use_yaml_template: bool = True,
        **kwargs
    ):
        """
        初始化V2分析器

        Args:
            analysis_type: 分析类型（mv_violation, ebike_violation, ads_presence, ads_behavior）
            consumer_group: Redis consumer group名称
            use_yaml_template: 是否使用YAML模板（默认True）
            **kwargs: 传递给基类的其他参数
        """
        if consumer_group is None:
            consumer_group = f"{analysis_type}_workers_v2"

        super().__init__(
            analysis_type=analysis_type,
            consumer_group=consumer_group,
            **kwargs
        )

        self.use_yaml_template = use_yaml_template
        self._prompt_content: Optional[str] = None
        self._embedding_fn_registered = False

    def get_templates(self) -> List[str]:
        """
        返回该分析类型的语义模板列表

        V2版本从YAML模板提取所有正样本文本作为向后兼容
        """
        if not self.use_yaml_template:
            # 子类可覆盖
            return []

        try:
            registry = get_template_registry()
            template = registry.get_template(self.analysis_type)
            if template is None:
                return []

            # 提取所有正样本作为模板
            all_positives = []
            for group in template.groups:
                all_positives.extend(group.positives)

            return all_positives

        except Exception as e:
            self.log.error(f"加载YAML模板失败: {e}")
            return []

    def get_vlm_prompt(self) -> str:
        """
        返回该分析类型的VLM prompt

        从AnalysisCatalog加载对应的prompt文件
        """
        if self._prompt_content is not None:
            return self._prompt_content

        try:
            from traffic_vlm.semantic import AnalysisType
            catalog = get_analysis_catalog()

            # 映射字符串到枚举
            type_mapping = {
                "mv_violation": AnalysisType.MV_VIOLATION,
                "ebike_violation": AnalysisType.EBIKE_VIOLATION,
                "ads_presence": AnalysisType.ADS_PRESENCE,
                "ads_behavior": AnalysisType.ADS_BEHAVIOR
            }

            analysis_enum = type_mapping.get(self.analysis_type)
            if analysis_enum is None:
                self.log.warning(f"未知分析类型: {self.analysis_type}")
                return self._get_fallback_prompt()

            prompt = catalog.load_prompt(analysis_enum)
            if prompt:
                self._prompt_content = prompt
                return prompt

        except Exception as e:
            self.log.error(f"加载prompt失败: {e}")

        return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """回退prompt（当无法加载时使用）"""
        return f"""分析视频帧，判断是否存在与 {self.analysis_type} 相关的内容。

请按以下JSON格式回答:
{{
    "judgment": "YES/NO/UNCERTAIN",
    "confidence": 0.0-1.0,
    "reason": "简要说明"
}}"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """
        解析VLM响应

        支持多种响应格式，根据analysis_type选择合适的解析策略
        """
        response_text = response_text.strip()

        # 尝试解析JSON
        json_match = re.search(r'\{[\s\S]*?\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._parse_json_result(data, response_text)
            except json.JSONDecodeError:
                pass

        # 回退到简单解析
        return self._parse_simple_response(response_text)

    def _parse_json_result(self, data: Dict[str, Any], raw_text: str) -> AnalysisResult:
        """解析JSON结果"""
        judgment = data.get("judgment", "UNCERTAIN").upper()
        confidence = float(data.get("confidence", 0.5))
        reason = data.get("reason", raw_text[:200])

        result = AnalysisResult(
            judgment=judgment,
            confidence=confidence,
            reason=reason,
            raw_response=raw_text
        )

        # 根据分析类型设置额外字段
        if self.analysis_type == "mv_violation":
            subtypes = data.get("subtypes", [])
            if subtypes:
                result.violation_type = subtypes[0] if isinstance(subtypes, list) else subtypes

        elif self.analysis_type == "ebike_violation":
            subtypes = data.get("subtypes", [])
            object_type = data.get("object_type", "ebike")
            if subtypes:
                result.violation_type = subtypes[0] if isinstance(subtypes, list) else subtypes
            result.extra["object_type"] = object_type

        elif self.analysis_type == "ads_presence":
            marker_light = data.get("marker_light", {})
            result.marker_light_state = "present" if marker_light.get("present") else "absent"
            result.extra["marker_light"] = marker_light
            result.extra["exclusion_checks"] = data.get("exclusion_checks", [])

        elif self.analysis_type == "ads_behavior":
            result.extra["control_source"] = data.get("control_source", "unknown")
            result.extra["tags"] = data.get("tags", [])

        return result

    def _parse_simple_response(self, response_text: str) -> AnalysisResult:
        """简单解析响应"""
        upper_text = response_text.upper()[:200]

        judgment = "UNCERTAIN"
        confidence = 0.5

        if "YES" in upper_text:
            judgment = "YES"
            confidence = 0.7
        elif "NO" in upper_text:
            judgment = "NO"
            confidence = 0.7

        return AnalysisResult(
            judgment=judgment,
            confidence=confidence,
            reason=response_text[:200],
            raw_response=response_text
        )

    def compute_similarity_scores_v2(
        self,
        frames: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> List[FrameInfo]:
        """
        V2版本的相似度计算

        使用YAML模板的三元组评分公式
        """
        if not self.use_yaml_template:
            # 回退到基类方法
            return super().compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

        try:
            registry = get_template_registry()

            # 注册嵌入函数（只做一次）
            if not self._embedding_fn_registered and registry.embedding_fn is None:
                registry.embedding_fn = self._get_embedding_fn()
                self._embedding_fn_registered = True

            for frame in frames:
                if frame.embedding is None:
                    continue

                score, meta = registry.compute_frame_score(
                    frame.embedding,
                    self.analysis_type,
                    return_all_groups=False
                )

                frame.similarity_score = score
                frame.template_hit = score > 0.3

                # 存储额外信息到帧
                if hasattr(frame, 'extra'):
                    frame.extra = meta
                else:
                    # FrameInfo没有extra字段，存到similarity_score相关
                    pass

            self.log.info(
                f"V2评分完成: {len(frames)} 帧",
                job_id=job_id,
                trace_id=trace_id
            )

        except Exception as e:
            self.log.error(f"V2评分失败，回退到基类方法: {e}")
            return super().compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

        return frames

    def _get_embedding_fn(self):
        """获取嵌入函数"""
        import requests
        import numpy as np

        def embed_texts(texts: List[str]) -> np.ndarray:
            resp = requests.post(
                f"{self.embedding_url}/encode/text",
                json={"texts": texts},
                timeout=60
            )
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"])

        return embed_texts

    def process_task(self, task) -> bool:
        """
        处理任务（覆盖基类方法）

        使用V2评分方法
        """
        import time

        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id

        self.log.info(
            f"[V2] 开始处理任务 [{self.analysis_type}]: {task.window_path}",
            job_id=job_id,
            trace_id=trace_id
        )

        try:
            self.redis.set_task_status(job_id, "processing", self.consumer_name)

            # 1. 抽帧
            frames = self.extract_frames(task.window_path, fps=1.0, job_id=job_id, trace_id=trace_id)
            if not frames:
                self.log.warning("抽帧为空，标记完成", job_id=job_id)
                self.redis.set_task_status(job_id, "done", self.consumer_name)
                return True

            # 2. 计算嵌入
            frames = self.compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

            # 3. V2相似度评分
            frames = self.compute_similarity_scores_v2(frames, job_id=job_id, trace_id=trace_id)

            # 4. 聚类
            clips = self.cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

            # 5. VLM分析
            analysis_result = AnalysisResult(
                judgment="NO",
                confidence=1.0,
                reason="无可疑片段"
            )

            if clips:
                best_clip = max(clips, key=lambda c: c.clip_score)

                if best_clip.clip_score >= CLIP_SCORE_THRESHOLD:
                    self.log.info(
                        f"[V2] 最高分片段 {best_clip.clip_score:.3f} >= 阈值 {CLIP_SCORE_THRESHOLD}, 调用 VLM",
                        job_id=job_id, trace_id=trace_id
                    )
                    keyframes = self.select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                    best_clip.keyframes = keyframes
                    analysis_result = self.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)
                else:
                    self.log.info(
                        f"[V2] 最高分片段 {best_clip.clip_score:.3f} < 阈值 {CLIP_SCORE_THRESHOLD}, 跳过 VLM",
                        job_id=job_id, trace_id=trace_id
                    )
                    analysis_result = AnalysisResult(
                        judgment="NO",
                        confidence=1.0,
                        reason=f"最高分 {best_clip.clip_score:.3f} 低于阈值"
                    )

            # 6. 保存结果
            processing_time = time.time() - start_time
            result_path = self.save_result(task, clips, analysis_result, processing_time)

            # 7. 发送结果任务
            from workers.common.redis_client import ResultTask
            result_task = ResultTask(
                event_time=task.seg_end_ts or time.time(),
                job_id=job_id,
                camera_id=task.camera_id,
                trace_id=trace_id,
                result_path=str(result_path),
                is_accident=False,  # 非事故分析
                is_positive=analysis_result.judgment == "YES",
                confidence=analysis_result.confidence,
                seg_end_ts=task.seg_end_ts,
                analysis_type=self.analysis_type,
                violation_type=analysis_result.violation_type,
                behavior_type=analysis_result.behavior_type
            )
            self.redis.add_result_task(result_task)

            # 8. 清理临时文件
            self._cleanup_frames(frames)

            # 9. 更新状态
            self.redis.set_task_status(job_id, "done", self.consumer_name)

            self.log.info(
                f"[V2] 任务完成: {job_id}, 耗时={processing_time:.1f}s, 判断={analysis_result.judgment}",
                job_id=job_id,
                trace_id=trace_id
            )

            return True

        except Exception as e:
            self.log.error(f"[V2] 任务处理失败: {e}", job_id=job_id, trace_id=trace_id)
            self.redis.set_task_status(job_id, "failed", self.consumer_name)
            return False


class ADSBehaviorAnalyzer(SemanticAnalyzerV2):
    """
    ADS行为分析器 - 带门控的两阶段分析

    特点：
    - 仅在ads_presence=YES时执行
    - 分析自动驾驶车辆的驾驶行为
    """

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="ads_behavior",
            **kwargs
        )

    def should_process_task(self, task) -> bool:
        """
        检查门控条件

        只有当ads_presence=YES时才处理
        """
        # 基本类型检查
        if not super().should_process_task(task):
            return False

        # 检查门控
        try:
            from traffic_vlm.semantic import check_ads_behavior_gate

            # 从任务元数据获取ads_presence结果
            ads_presence_result = getattr(task, 'ads_presence_result', None)
            if ads_presence_result is None:
                # 尝试从Redis获取
                ads_presence_result = self._get_ads_presence_result(task.job_id)

            if ads_presence_result is None:
                self.log.info(
                    f"无ads_presence结果，跳过ads_behavior分析",
                    job_id=task.job_id
                )
                return False

            if not check_ads_behavior_gate(ads_presence_result):
                self.log.info(
                    f"ADS门控未通过: ads_presence={ads_presence_result.get('judgment')}",
                    job_id=task.job_id
                )
                return False

            return True

        except Exception as e:
            self.log.error(f"门控检查失败: {e}")
            return False

    def _get_ads_presence_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """从Redis获取ads_presence结果"""
        try:
            key = f"ads_presence:{job_id}"
            data = self.redis.client.get(key)
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None


# 分析器工厂
def create_analyzer(analysis_type: str, **kwargs) -> SemanticAnalyzerV2:
    """
    创建分析器实例

    Args:
        analysis_type: 分析类型
        **kwargs: 分析器参数

    Returns:
        SemanticAnalyzerV2 实例
    """
    if analysis_type == "ads_behavior":
        return ADSBehaviorAnalyzer(**kwargs)
    else:
        return SemanticAnalyzerV2(analysis_type=analysis_type, **kwargs)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Semantic Analyzer V2")
    parser.add_argument("--type", required=True, choices=["mv_violation", "ebike_violation", "ads_presence", "ads_behavior"],
                       help="分析类型")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    analyzer = create_analyzer(
        analysis_type=args.type,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
