#!/usr/bin/env python3
"""
自动驾驶行为检测分析器 (ads_behavior)

检测目标:
- 示廓灯开启/关闭状态
- 自动驾驶模式下的异常行为
- 车辆行为模式分析

特点:
- 两阶段 VLM 调用:
  1. 第一阶段: 检测示廓灯是否存在及状态
  2. 第二阶段: 分析示廓灯开启时的车辆行为
- 专门针对自动驾驶测试车辆的行为分析
"""
import os
import sys
import re
import time
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from workers.semantic_base import (
    SemanticAnalyzerBase,
    AnalysisResult,
    FrameInfo,
    ClipInfo,
    CLIP_SCORE_THRESHOLD
)


# 自动驾驶行为语义模板
ADS_BEHAVIOR_TEMPLATES = [
    # 示廓灯/标识灯
    "vehicle with marker lights on",
    "car with side lights illuminated",
    "autonomous vehicle indicator lights",
    "self-driving car with lights on",
    "AV marker lights visible",

    # 自动驾驶车辆特征
    "autonomous vehicle on road",
    "self-driving car",
    "vehicle with sensors on roof",
    "car with lidar sensor",

    # 行为模式
    "vehicle stopping suddenly",
    "car making unusual turn",
    "vehicle changing speed abruptly",
    "car taking unusual path",
    "vehicle in autonomous mode",

    # 交互场景
    "autonomous car at intersection",
    "self-driving vehicle near pedestrians",
    "AV interacting with traffic"
]


class ADSBehaviorAnalyzer(SemanticAnalyzerBase):
    """自动驾驶行为检测分析器"""

    def __init__(self, **kwargs):
        super().__init__(
            analysis_type="ads_behavior",
            consumer_group="ads_behavior_workers",
            **kwargs
        )

    def get_templates(self) -> List[str]:
        return ADS_BEHAVIOR_TEMPLATES

    def get_vlm_prompt(self) -> str:
        """第一阶段 prompt: 检测示廓灯"""
        return """分析这些视频帧，检测是否存在开启示廓灯/标识灯的车辆。

示廓灯特征:
- 通常位于车辆顶部或车身两侧
- 可能是自动驾驶测试车辆的标识灯
- 颜色可能为黄色、白色或其他醒目颜色
- 在夜间或低光照环境下更明显

请按以下 JSON 格式回答:
{
    "marker_light_detected": true/false,
    "marker_light_state": "开启/关闭/不确定/无",
    "vehicle_count": 0-N,
    "confidence": 0.0-1.0,
    "reason": "简要说明判断依据"
}

注意事项:
- 仔细观察车顶和车身是否有额外灯光
- 区分普通车灯和特殊标识灯
- 注意多辆车的情况"""

    def get_behavior_analysis_prompt(self) -> str:
        """第二阶段 prompt: 分析行为"""
        return """已检测到开启示廓灯的车辆。请分析该车辆的驾驶行为。

请检测以下行为:
1. 变道行为 - 是否存在变道，变道是否平稳
2. 转弯行为 - 转弯角度、速度控制
3. 加减速 - 是否存在急加速或急减速
4. 与周围车辆的交互 - 安全距离、礼让行为
5. 行驶轨迹 - 是否保持车道居中
6. 异常行为 - 是否存在任何不寻常的驾驶模式

请按以下 JSON 格式回答:
{
    "judgment": "YES/NO/UNCERTAIN",
    "confidence": 0.0-1.0,
    "behavior_type": "正常行驶/变道/转弯/加减速/异常行为/无法判断",
    "behavior_description": "详细描述观察到的行为",
    "safety_level": "安全/注意/危险",
    "reason": "行为分析依据"
}

注意事项:
- 关注示廓灯车辆的具体行为
- 分析其与周围交通参与者的交互
- 评估行为的安全性"""

    def parse_vlm_response(self, response_text: str) -> AnalysisResult:
        """解析第一阶段 VLM 响应（示廓灯检测）"""
        import json

        response_text = response_text.strip()
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                marker_detected = data.get("marker_light_detected", False)
                marker_state_raw = data.get("marker_light_state", "无")

                # 规范化状态（支持中英文）
                marker_state = marker_state_raw
                if marker_state_raw.lower() in ("on", "开启", "开"):
                    marker_state = "开启"
                elif marker_state_raw.lower() in ("off", "关闭", "关"):
                    marker_state = "关闭"

                # 如果检测到示廓灯开启，标记为需要第二阶段分析
                judgment = "YES" if marker_detected and marker_state == "开启" else "NO"

                return AnalysisResult(
                    judgment=judgment,
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", response_text),
                    marker_light_state=marker_state,
                    extra={"vehicle_count": data.get("vehicle_count", 0)}
                )
            except json.JSONDecodeError:
                pass

        # 回退解析
        judgment = "UNCERTAIN"
        confidence = 0.5
        marker_state = "不确定"

        if "true" in response_text.lower() or "开启" in response_text:
            judgment = "YES"
            confidence = 0.7
            marker_state = "开启"
        elif "false" in response_text.lower() or "关闭" in response_text or "无" in response_text:
            judgment = "NO"
            confidence = 0.7
            marker_state = "关闭"

        return AnalysisResult(
            judgment=judgment,
            confidence=confidence,
            reason=response_text,
            marker_light_state=marker_state
        )

    def parse_behavior_response(self, response_text: str) -> AnalysisResult:
        """解析第二阶段 VLM 响应（行为分析）"""
        import json

        response_text = response_text.strip()
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)

        if json_match:
            try:
                data = json.loads(json_match.group())
                return AnalysisResult(
                    judgment=data.get("judgment", "UNCERTAIN"),
                    confidence=float(data.get("confidence", 0.5)),
                    reason=data.get("reason", response_text),
                    behavior_type=data.get("behavior_type"),
                    extra={
                        "behavior_description": data.get("behavior_description"),
                        "safety_level": data.get("safety_level")
                    }
                )
            except json.JSONDecodeError:
                pass

        return AnalysisResult(
            judgment="UNCERTAIN",
            confidence=0.5,
            reason=response_text
        )

    def call_vlm_phase2(
        self,
        keyframes: List[FrameInfo],
        job_id: str = "",
        trace_id: str = ""
    ) -> AnalysisResult:
        """第二阶段 VLM 调用: 行为分析"""
        import base64
        import requests

        if not keyframes:
            return AnalysisResult(
                judgment="NO",
                confidence=0.0,
                reason="无关键帧"
            )

        images_b64 = []
        for frame in keyframes:
            try:
                with open(frame.frame_path, "rb") as f:
                    img_bytes = f.read()
                images_b64.append(base64.b64encode(img_bytes).decode())
            except Exception as e:
                self.log.warning(f"读取关键帧失败: {frame.frame_path}, {e}")

        if not images_b64:
            return AnalysisResult(
                judgment="NO",
                confidence=0.0,
                reason="无法读取关键帧"
            )

        prompt = self.get_behavior_analysis_prompt()

        content = [{"type": "text", "text": prompt}]
        for b64_img in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })

        try:
            resp = requests.post(
                f"{self.vlm_proxy_url}/chat/completions",
                json={
                    "model": "qwen3-vl-32b",
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 800,
                    "temperature": 0.1
                },
                headers={
                    "X-Trace-Id": trace_id,
                    "X-Job-Id": job_id,
                    "X-Phase": "behavior-analysis"
                },
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()

            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            result = self.parse_behavior_response(response_text)
            result.raw_response = response_text

            self.log.info(
                f"Phase2 VLM 行为分析: behavior_type={result.behavior_type}",
                job_id=job_id,
                trace_id=trace_id
            )

            return result

        except Exception as e:
            self.log.error(f"Phase2 VLM 调用失败: {e}", job_id=job_id, trace_id=trace_id)
            return AnalysisResult(
                judgment="ERROR",
                confidence=0.0,
                reason=str(e)
            )

    def process_task(self, task):
        """覆盖父类方法，实现两阶段 VLM 调用"""
        from workers.common.redis_client import ResultTask

        start_time = time.time()
        job_id = task.job_id
        trace_id = task.trace_id

        self.log.info(
            f"开始处理任务 [{self.analysis_type}]: {task.window_path}",
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

            # 3. 计算相似度
            frames = self.compute_similarity_scores(frames, job_id=job_id, trace_id=trace_id)

            # 4. 聚类
            clips = self.cluster_frames_to_clips(frames, job_id=job_id, trace_id=trace_id)

            # 5. 两阶段 VLM 分析
            analysis_result = AnalysisResult(
                judgment="NO",
                confidence=1.0,
                reason="无可疑片段",
                marker_light_state="无"
            )

            if clips:
                best_clip = max(clips, key=lambda c: c.clip_score)

                if best_clip.clip_score >= CLIP_SCORE_THRESHOLD:
                    keyframes = self.select_keyframes(best_clip, max_frames=12, job_id=job_id, trace_id=trace_id)
                    best_clip.keyframes = keyframes

                    # Phase 1: 示廓灯检测
                    self.log.info("Phase1: 示廓灯检测", job_id=job_id, trace_id=trace_id)
                    phase1_result = self.call_vlm(keyframes, job_id=job_id, trace_id=trace_id)

                    if phase1_result.judgment == "YES" and phase1_result.marker_light_state == "开启":
                        # Phase 2: 行为分析
                        self.log.info("Phase2: 行为分析（检测到示廓灯开启）", job_id=job_id, trace_id=trace_id)
                        phase2_result = self.call_vlm_phase2(keyframes, job_id=job_id, trace_id=trace_id)

                        # 合并结果
                        analysis_result = AnalysisResult(
                            judgment=phase2_result.judgment,
                            confidence=phase2_result.confidence,
                            reason=phase2_result.reason,
                            marker_light_state=phase1_result.marker_light_state,
                            behavior_type=phase2_result.behavior_type,
                            extra={
                                "phase1": {
                                    "marker_light_state": phase1_result.marker_light_state,
                                    "confidence": phase1_result.confidence
                                },
                                "phase2": phase2_result.extra
                            }
                        )
                    else:
                        # 未检测到示廓灯，使用 Phase 1 结果
                        analysis_result = phase1_result
                else:
                    analysis_result = AnalysisResult(
                        judgment="NO",
                        confidence=1.0,
                        reason=f"最高分 {best_clip.clip_score:.3f} 低于阈值",
                        marker_light_state="无"
                    )

            # 6. 保存结果
            processing_time = time.time() - start_time
            result_path = self.save_result(task, clips, analysis_result, processing_time)

            # 7. 发送结果任务
            result_task = ResultTask(
                event_time=task.seg_end_ts or time.time(),
                job_id=job_id,
                camera_id=task.camera_id,
                trace_id=trace_id,
                result_path=str(result_path),
                is_accident=False,  # ads_behavior 不是事故检测
                is_positive=analysis_result.judgment == "YES",
                confidence=analysis_result.confidence,
                seg_end_ts=task.seg_end_ts,
                analysis_type=self.analysis_type,
                behavior_type=analysis_result.behavior_type
            )
            self.redis.add_result_task(result_task)

            # 8. 清理
            self._cleanup_frames(frames)

            # 9. 更新状态
            self.redis.set_task_status(job_id, "done", self.consumer_name)

            self.log.info(
                f"任务完成: {job_id}, 耗时={processing_time:.1f}s, "
                f"marker_light={analysis_result.marker_light_state}, "
                f"behavior={analysis_result.behavior_type}",
                job_id=job_id,
                trace_id=trace_id
            )

            return True

        except Exception as e:
            self.log.error(f"任务处理失败: {e}", job_id=job_id, trace_id=trace_id)
            self.redis.set_task_status(job_id, "failed", self.consumer_name)
            return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="自动驾驶行为检测分析器")
    parser.add_argument("--redis-host", default=os.getenv("REDIS_HOST", "localhost"))
    parser.add_argument("--redis-port", type=int, default=int(os.getenv("REDIS_PORT", "6379")))
    parser.add_argument("--embedding-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8080"))
    parser.add_argument("--vlm-proxy-url", default=os.getenv("VLM_PROXY_URL", "http://localhost:8001"))
    parser.add_argument("--results-dir", default=os.getenv("RESULTS_DIR", "/data1/results"))
    args = parser.parse_args()

    analyzer = ADSBehaviorAnalyzer(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        embedding_url=args.embedding_url,
        vlm_proxy_url=args.vlm_proxy_url,
        results_dir=args.results_dir
    )
    analyzer.run()


if __name__ == "__main__":
    main()
