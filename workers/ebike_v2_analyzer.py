#!/usr/bin/env python3
"""
Ebike Violation V2 Analyzer - 二轮车违法检测分析器

特性：
1. 使用margin + Top-K打分策略
2. Feature Flag控制（EBIKE_V2_SCORING）
3. 不影响accident pipeline
4. 结构化日志输出

使用方式：
    analyzer = EbikeV2Analyzer(
        redis_host="localhost",
        embedding_url="http://localhost:8080",
        vlm_proxy_url="http://localhost:8001"
    )
    analyzer.run()
"""

import os
import sys
import time
import json
import re
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
import numpy as np

from workers.semantic_base import (
    SemanticAnalyzerBase,
    AnalysisResult,
    FrameInfo,
    ClipInfo,
    CLIP_SCORE_THRESHOLD
)
from workers.ebike_v2_scorer import (
    EbikeV2Scorer,
    EBIKE_V2_ENABLED,
    EBIKE_V2_TOP_M,
    EBIKE_V2_COOLDOWN_SEC,
    log_ebike_v2_result
)


class EbikeV2Analyzer(SemanticAnalyzerBase):
    """
    Ebike V2 分析器

    当 EBIKE_V2_SCORING=true 时使用 margin + Top-K 策略
    当 EBIKE_V2_SCORING=false 时回退到基类逻辑（保持兼容）

    Gate 特性（仅 EBIKE_V2_SCORING=true 时生效）：
    - Top-M Clip Gate: 只选择 Top M 个 clips 送 VLM（EBIKE_V2_TOP_M）
    - Camera Cooldown Gate: 同一 camera_id 冷却期内最多触发 1 次（EBIKE_V2_COOLDOWN_SEC）
    """

    # Cooldown Redis key 前缀
    COOLDOWN_KEY_PREFIX = "cooldown:ebike:"

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="ebike_violation",
            consumer_group="ebike_violation_workers",
            **kwargs
        )

        self._v2_scorer: Optional[EbikeV2Scorer] = None
        self._v2_prompt: Optional[str] = None
        self._embedding_fn_ready = False

        # 记录门控配置（用于日志）
        self.log.info(
            f"EbikeV2Analyzer initialized: V2={EBIKE_V2_ENABLED}, "
            f"TOP_M={EBIKE_V2_TOP_M}, COOLDOWN_SEC={EBIKE_V2_COOLDOWN_SEC}"
        )

    def _check_cooldown(self, camera_id: str, job_id: str, trace_id: str) -> Tuple[bool, str]:
        """
        检查并尝试设置 camera 冷却期

        Returns:
            (is_cooled_down, reason): is_cooled_down=True 表示在冷却期内应跳过
        """
        if not EBIKE_V2_ENABLED or EBIKE_V2_COOLDOWN_SEC <= 0:
            return False, ""

        cooldown_key = f"{self.COOLDOWN_KEY_PREFIX}{camera_id}"

        try:
            # SET NX EX: 仅当 key 不存在时设置，并设置过期时间
            # 返回 True 表示设置成功（不在冷却期），返回 False/None 表示已存在（在冷却期）
            result = self.redis.client.set(
                cooldown_key,
                job_id,  # 存储触发此冷却的 job_id
                nx=True,
                ex=EBIKE_V2_COOLDOWN_SEC
            )

            if result:
                # 设置成功，不在冷却期
                self.log.info(
                    f"Cooldown set: {cooldown_key} (TTL={EBIKE_V2_COOLDOWN_SEC}s)",
                    job_id=job_id, trace_id=trace_id
                )
                return False, ""
            else:
                # 已存在，在冷却期内
                ttl = self.redis.client.ttl(cooldown_key)
                reason = f"camera {camera_id} in cooldown (TTL={ttl}s)"
                self.log.info(
                    f"Cooldown active: {reason}",
                    job_id=job_id, trace_id=trace_id
                )
                return True, reason

        except Exception as e:
            self.log.warning(f"Cooldown check failed: {e}", job_id=job_id, trace_id=trace_id)
            return False, ""  # 失败时不阻止，降级为无冷却

    def _get_embedding_fn(self):
        """获取嵌入函数"""
        def embed_texts(texts: List[str]) -> np.ndarray:
            resp = requests.post(
                f"{self.embedding_url}/encode/text",
                json={"texts": texts},
                timeout=60
            )
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"])
        return embed_texts

    def _init_v2_scorer(self):
        """初始化V2打分器"""
        if not EBIKE_V2_ENABLED:
            return

        if self._v2_scorer is not None:
            return

        try:
            embed_fn = self._get_embedding_fn()
            self._v2_scorer = EbikeV2Scorer(embed_fn)
            self.log.info("EbikeV2Scorer initialized")
        except Exception as e:
            self.log.error(f"Failed to init EbikeV2Scorer: {e}")

    def get_templates(self) -> List[str]:
        """返回模板列表（V1兼容）"""
        # V2模式下仍然加载基本模板用于回退
        return [
            "electric scooter riding in car lane",
            "e-bike running red light intersection",
            "two-wheeler wrong way driving",
            "e-bike rider without helmet",
            "路口电动车违法",
            "二轮车闯红灯",
            "电动车逆行",
        ]

    def get_vlm_prompt(self) -> str:
        """返回VLM Prompt"""
        if EBIKE_V2_ENABLED:
            return self._load_v2_prompt()
        else:
            return self._get_v1_prompt()

    def _load_v2_prompt(self) -> str:
        """加载V2 Prompt"""
        if self._v2_prompt is not None:
            return self._v2_prompt

        prompt_path = Path(PROJECT_ROOT) / "traffic_vlm/prompts/ebike_violation_v2_prompt.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                self._v2_prompt = f.read()
            return self._v2_prompt
        except Exception as e:
            self.log.error(f"Failed to load V2 prompt: {e}")
            return self._get_v1_prompt()

    def _get_v1_prompt(self) -> str:
        """V1 Prompt（回退）"""
        return """分析视频帧，判断是否存在电动自行车/二轮车交通违法行为。

检测类型：
1. 占用机动车道
2. 闯红灯
3. 逆行
4. 未戴头盔/违法载人

请按JSON格式回答：
{
    "judgment": "YES/NO/UNCERTAIN",
    "confidence": 0.0-1.0,
    "violation_type": "占用机动车道/闯红灯/逆行/头盔载人/无",
    "reason": "简要说明"
}"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析VLM响应"""
        response_text = response_text.strip()

        # 尝试解析JSON
        json_match = re.search(r'\{[\s\S]*?\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group())

                # V2格式
                if "violation" in data:
                    judgment = data.get("violation", "UNCERTAIN").upper()
                    subtype = data.get("subtype", "unknown")
                    object_type = data.get("object_type", "unknown")

                    return AnalysisResult(
                        judgment=judgment,
                        confidence=float(data.get("confidence", 0.5)),
                        reason=data.get("reason", response_text[:200]),
                        violation_type=subtype,
                        extra={
                            "object_type": object_type,
                            "evidence": data.get("evidence", {}),
                            "uncertain_reason": data.get("uncertain_reason", "")
                        }
                    )

                # V1格式兼容
                return AnalysisResult(
                    judgment=data.get("judgment", "UNCERTAIN").upper(),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", response_text[:200]),
                    violation_type=data.get("violation_type")
                )

            except json.JSONDecodeError:
                pass

        # 回退解析
        judgment = "UNCERTAIN"
        if "YES" in response_text.upper()[:100]:
            judgment = "YES"
        elif "NO" in response_text.upper()[:100]:
            judgment = "NO"

        return AnalysisResult(
            judgment=judgment,
            confidence=0.5,
            reason=response_text[:200]
        )

    def process_task(self, task) -> bool:
        """处理任务"""
        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id
        camera_id = task.camera_id

        self.log.info(
            f"[EbikeV2={EBIKE_V2_ENABLED}] 开始处理: {task.window_path}",
            job_id=job_id, trace_id=trace_id
        )

        try:
            self.redis.set_task_status(job_id, "processing", self.consumer_name)

            # 1. 抽帧
            frames = self.extract_frames(task.window_path, fps=1.0, job_id=job_id, trace_id=trace_id)
            if not frames:
                self.log.warning("抽帧为空", job_id=job_id)
                self.redis.set_task_status(job_id, "done", self.consumer_name)
                return True

            # 2. 计算嵌入
            frames = self.compute_embeddings(frames, job_id=job_id, trace_id=trace_id)

            # 3. 根据Feature Flag选择打分策略
            if EBIKE_V2_ENABLED:
                clips, analysis_result, vlm_calls, vlm_time = self._process_v2(
                    frames, job_id, trace_id, camera_id
                )
            else:
                clips, analysis_result, vlm_calls, vlm_time = self._process_v1(
                    frames, job_id, trace_id
                )

            # 4. 保存结果
            processing_time = time.time() - start_time
            result_path = self.save_result(task, clips, analysis_result, processing_time)

            # 5. 发送结果任务
            from workers.common.redis_client import ResultTask
            result_task = ResultTask(
                event_time=task.seg_end_ts or time.time(),
                job_id=job_id,
                camera_id=camera_id,
                trace_id=trace_id,
                result_path=str(result_path),
                is_accident=False,
                is_positive=analysis_result.judgment == "YES",
                confidence=analysis_result.confidence,
                seg_end_ts=task.seg_end_ts,
                analysis_type=self.analysis_type,
                violation_type=analysis_result.violation_type
            )
            self.redis.add_result_task(result_task)

            # 6. 清理
            self._cleanup_frames(frames)
            self.redis.set_task_status(job_id, "done", self.consumer_name)

            self.log.info(
                f"任务完成: {job_id}, 耗时={processing_time:.1f}s, "
                f"判断={analysis_result.judgment}, VLM调用={vlm_calls}",
                job_id=job_id, trace_id=trace_id
            )

            return True

        except Exception as e:
            self.log.error(f"任务失败: {e}", job_id=job_id, trace_id=trace_id)
            self.redis.set_task_status(job_id, "failed", self.consumer_name)
            return False

    def _process_v1(self, frames, job_id, trace_id):
        """V1处理流程（原有逻辑）"""
        # 计算相似度
        frames = self.compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

        # 聚类
        clips = self.cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

        # VLM分析
        analysis_result = AnalysisResult(
            judgment="NO",
            confidence=1.0,
            reason="无可疑片段"
        )
        vlm_calls = 0
        vlm_time = 0

        if clips:
            best_clip = max(clips, key=lambda c: c.clip_score)
            if best_clip.clip_score >= CLIP_SCORE_THRESHOLD:
                keyframes = self.select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                vlm_start = time.time()
                analysis_result = self.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)
                vlm_time = (time.time() - vlm_start) * 1000
                vlm_calls = 1
            else:
                analysis_result = AnalysisResult(
                    judgment="NO",
                    confidence=1.0,
                    reason=f"最高分 {best_clip.clip_score:.3f} 低于阈值"
                )

        # 转换clips格式
        clip_list = [
            ClipInfo(
                clip_id=c.clip_id,
                start_sec=c.start_sec,
                end_sec=c.end_sec,
                frames=c.frames,
                clip_score=c.clip_score
            ) for c in clips
        ] if clips else []

        return clip_list, analysis_result, vlm_calls, vlm_time

    def _process_v2(self, frames, job_id, trace_id, camera_id):
        """V2处理流程（margin + Top-K + 门控）"""
        # 初始化V2打分器
        self._init_v2_scorer()

        if self._v2_scorer is None:
            self.log.warning("V2 scorer not available, fallback to V1")
            return self._process_v1(frames, job_id, trace_id)

        # 计算margin分数
        frame_scores = self._v2_scorer.compute_margin_scores(frames, job_id, trace_id)

        # 选择Top-K帧
        topk_scores = self._v2_scorer.select_topk_frames(frame_scores)

        # 获取统计信息
        stats = self._v2_scorer.get_stats(topk_scores)

        # 聚类
        clips = self._v2_scorer.cluster_frames_to_clips(topk_scores, frames)

        # ==================== 门控增强结果结构 ====================
        ebike_v2_meta = {
            "enabled": True,
            "top_m": EBIKE_V2_TOP_M,
            "cooldown_sec": EBIKE_V2_COOLDOWN_SEC,
            "vlm_called": False,
            "vlm_skipped_reason": None,
            "clips_total": len(clips),
            "clips_sent_to_vlm": 0,
            "max_margin": stats.get("margin_max") if stats else None
        }

        # VLM分析
        analysis_result = AnalysisResult(
            judgment="NO",
            confidence=1.0,
            reason="无可疑片段"
        )
        vlm_calls = 0
        vlm_time = 0

        if clips:
            # ==================== Gate A: Top-M Clip Gate ====================
            # 按 clip_score 降序排序，只选择 Top M 个 clips
            sorted_clips = sorted(clips, key=lambda c: c["clip_score"], reverse=True)
            clips_to_process = sorted_clips[:EBIKE_V2_TOP_M]
            ebike_v2_meta["clips_sent_to_vlm"] = len(clips_to_process)

            self.log.info(
                f"V2 Gate A: Top-M={EBIKE_V2_TOP_M}, clips_total={len(clips)}, "
                f"clips_sent={len(clips_to_process)}",
                job_id=job_id, trace_id=trace_id
            )

            # 选择最高分片段
            best_clip = clips_to_process[0]

            # ==================== Gate B: Camera Cooldown Gate ====================
            is_cooled_down, cooldown_reason = self._check_cooldown(camera_id, job_id, trace_id)

            if is_cooled_down:
                # 在冷却期内，跳过 VLM 调用
                ebike_v2_meta["vlm_skipped_reason"] = cooldown_reason
                analysis_result = AnalysisResult(
                    judgment="SKIPPED_COOLDOWN",
                    confidence=0.0,
                    reason=cooldown_reason,
                    extra={"ebike_v2": ebike_v2_meta}
                )
                self.log.info(
                    f"V2 Gate B: VLM skipped due to cooldown - {cooldown_reason}",
                    job_id=job_id, trace_id=trace_id
                )
            else:
                # 不在冷却期，调用 VLM
                self.log.info(
                    f"V2: 选中clip, margin_max={best_clip['clip_score']:.3f}, "
                    f"group={best_clip['best_group']}, frames={best_clip['frame_count']}",
                    job_id=job_id, trace_id=trace_id
                )

                # 获取关键帧
                keyframes = self._select_v2_keyframes(best_clip, frames, max_frames=12)

                if keyframes:
                    vlm_start = time.time()
                    analysis_result = self.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)
                    vlm_time = (time.time() - vlm_start) * 1000
                    vlm_calls = 1
                    ebike_v2_meta["vlm_called"] = True
        else:
            self.log.info("V2: 无候选clip", job_id=job_id, trace_id=trace_id)
            ebike_v2_meta["vlm_skipped_reason"] = "no_candidate_clips"

        # 将 ebike_v2 元数据添加到结果
        if not analysis_result.extra:
            analysis_result.extra = {}
        analysis_result.extra["ebike_v2"] = ebike_v2_meta

        # 输出结构化日志
        log_ebike_v2_result(
            self.log.logger,
            job_id=job_id,
            trace_id=trace_id,
            camera_id=camera_id,
            stats=stats,
            clips=clips,
            vlm_calls=vlm_calls,
            vlm_time_ms=vlm_time
        )

        # 输出门控汇总日志（JSON 格式便于日志聚合）
        gate_log = {
            "event": "ebike_v2_gates_summary",
            "job_id": job_id,
            "trace_id": trace_id,
            "camera_id": camera_id,
            "analysis_type": "ebike_violation",
            "gates": ebike_v2_meta,
            "vlm_calls": vlm_calls,
            "vlm_time_ms": vlm_time,
            "judgment": analysis_result.judgment
        }
        self.log.info(f"EBIKE_V2_GATES: {json.dumps(gate_log, ensure_ascii=False)}")

        # 转换clips格式
        clip_list = [
            ClipInfo(
                clip_id=c["clip_id"],
                start_sec=c["start_sec"],
                end_sec=c["end_sec"],
                frames=c.get("frames", []),
                clip_score=c["clip_score"]
            ) for c in clips
        ]

        return clip_list, analysis_result, vlm_calls, vlm_time

    def _select_v2_keyframes(self, clip: Dict, original_frames: List, max_frames: int = 12) -> List[FrameInfo]:
        """V2关键帧选择"""
        frame_scores = clip.get("frame_scores", [])
        if not frame_scores:
            return []

        # 按margin排序
        sorted_scores = sorted(frame_scores, key=lambda x: x.margin, reverse=True)

        keyframes = []
        for score in sorted_scores[:max_frames]:
            idx = score.frame_idx
            if idx < len(original_frames):
                keyframes.append(original_frames[idx])

        return keyframes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ebike Violation V2 Analyzer")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    print(f"EBIKE_V2_SCORING = {EBIKE_V2_ENABLED}")

    analyzer = EbikeV2Analyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
