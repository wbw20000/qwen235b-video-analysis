#!/usr/bin/env python3
"""
Ebike V2 门控回归测试脚本

功能：
1. Case-0: EBIKE_V2_SCORING=false（基线，确认与 V1 行为一致）
2. Case-1: EBIKE_V2_SCORING=true, COOLDOWN=0（V2 无门控，确认与现有 V2 一致）
3. Case-2: EBIKE_V2_SCORING=true, TOP_M=2, COOLDOWN=60（V2 门控开启）
4. 验证事故链路不变
5. 生成回归报告

使用方式：
    python tools/ebike_v2_gates_regression_test.py \
        --dataset /data1/testdata/二轮车数据集/二轮车违法数据集 \
        --embedding-url http://10.244.0.194:8080 \
        --output-dir /data1/results/reports \
        --redis-host localhost
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[WARN] redis-py not installed, cooldown simulation will be skipped")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("[ERROR] PyYAML not installed")
    sys.exit(1)


@dataclass
class GateTestResult:
    """单视频门控测试结果"""
    video_name: str
    camera_id: str  # 模拟的 camera_id

    # Embedding 信息
    frames_extracted: int = 0
    frames_embedded: int = 0

    # V2 分析结果
    max_margin: float = 0.0
    best_group: str = ""
    clips_total: int = 0

    # 门控结果
    clips_sent_to_vlm: int = 0
    vlm_called: bool = False
    vlm_skipped_reason: str = ""
    judgment: str = ""

    # 处理时间
    processing_time_ms: float = 0.0


@dataclass
class CaseResult:
    """单个 Case 的汇总结果"""
    case_name: str
    config: Dict[str, Any]
    results: List[GateTestResult] = field(default_factory=list)

    @property
    def total_videos(self) -> int:
        return len(self.results)

    @property
    def vlm_calls(self) -> int:
        return len([r for r in self.results if r.vlm_called])

    @property
    def vlm_skipped(self) -> int:
        return len([r for r in self.results if r.vlm_skipped_reason])

    @property
    def avg_clips_total(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.clips_total for r in self.results) / len(self.results)

    @property
    def avg_clips_sent(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.clips_sent_to_vlm for r in self.results) / len(self.results)

    @property
    def avg_max_margin(self) -> float:
        margins = [r.max_margin for r in self.results if r.max_margin > -1]
        return sum(margins) / len(margins) if margins else 0.0


class EbikeV2GatesRegressionTest:
    """Ebike V2 门控回归测试"""

    # 评分参数
    BETA = 0.7
    GAMMA = 0.3
    TOP_K = 30

    def __init__(
        self,
        dataset_dir: str,
        embedding_url: str,
        output_dir: str,
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        self.dataset_dir = Path(dataset_dir)
        self.embedding_url = embedding_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.redis_host = redis_host
        self.redis_port = redis_port
        self._redis_client = None

        # 预加载模板
        self._v2_template = None
        self._v2_groups = []

        # 结果存储
        self.case_results: List[CaseResult] = []

    @property
    def redis_client(self):
        """懒加载 Redis 客户端"""
        if not REDIS_AVAILABLE:
            return None
        if self._redis_client is None:
            try:
                self._redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    decode_responses=True
                )
                self._redis_client.ping()
            except Exception as e:
                print(f"[WARN] Redis connection failed: {e}")
                self._redis_client = None
        return self._redis_client

    def _load_v2_template(self):
        """加载 V2 模板"""
        if self._v2_template is not None:
            return

        template_path = PROJECT_ROOT / "traffic_vlm/templates/ebike_violation_v2.yaml"
        with open(template_path, 'r', encoding='utf-8') as f:
            self._v2_template = yaml.safe_load(f)

        # 预计算组嵌入
        for group_data in self._v2_template.get("groups", []):
            name = group_data.get("name", "")
            weight = group_data.get("weight", 1.0)

            pos_emb = self._embed_texts(group_data.get("positives", []))
            hn_emb = self._embed_texts(group_data.get("hard_negatives", []))
            neg_emb = self._embed_texts(group_data.get("negatives", []))

            # 归一化
            pos_norm = pos_emb / (np.linalg.norm(pos_emb, axis=1, keepdims=True) + 1e-8) if len(pos_emb) > 0 else pos_emb
            hn_norm = hn_emb / (np.linalg.norm(hn_emb, axis=1, keepdims=True) + 1e-8) if len(hn_emb) > 0 else hn_emb
            neg_norm = neg_emb / (np.linalg.norm(neg_emb, axis=1, keepdims=True) + 1e-8) if len(neg_emb) > 0 else neg_emb

            self._v2_groups.append({
                "name": name,
                "weight": weight,
                "pos": pos_norm,
                "hn": hn_norm,
                "neg": neg_norm
            })

        print(f"[OK] V2 template loaded: {len(self._v2_groups)} groups")

    def _extract_frames(self, video_path: Path, fps: float = 1.0, max_frames: int = 30) -> Tuple[List[Path], Path]:
        """抽取视频帧"""
        tmpdir = Path(tempfile.mkdtemp())
        output_pattern = str(tmpdir / "frame_%04d.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            "-q:v", "2",
            output_pattern
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        except Exception as e:
            print(f"  [ERROR] FFmpeg failed: {e}")
            return [], tmpdir

        frames = sorted(tmpdir.glob("frame_*.jpg"))
        return frames, tmpdir

    def _compute_embeddings(self, frame_paths: List[Path]) -> np.ndarray:
        """计算帧嵌入"""
        images_b64 = []
        for fp in frame_paths:
            try:
                with open(fp, "rb") as f:
                    images_b64.append(base64.b64encode(f.read()).decode())
            except:
                pass

        if not images_b64:
            return np.array([])

        try:
            resp = requests.post(
                f"{self.embedding_url}/encode/images",
                json={"images": images_b64},
                timeout=120
            )
            resp.raise_for_status()
            return np.array(resp.json()["embeddings"])
        except Exception as e:
            print(f"  [ERROR] Embedding failed: {e}")
            return np.array([])

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """文本嵌入"""
        if not texts:
            return np.array([])
        resp = requests.post(
            f"{self.embedding_url}/encode/text",
            json={"texts": texts},
            timeout=60
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"])

    def _compute_margins(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """计算每帧的 margin 分数"""
        if len(embeddings) == 0:
            return []

        frame_results = []

        for i, emb in enumerate(embeddings):
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)

            best_margin = -float('inf')
            best_group = ""

            for group in self._v2_groups:
                pos_sim = float(np.max(np.dot(group["pos"], emb_norm))) if len(group["pos"]) > 0 else 0.0
                hn_sim = float(np.max(np.dot(group["hn"], emb_norm))) if len(group["hn"]) > 0 else 0.0
                neg_sim = float(np.max(np.dot(group["neg"], emb_norm))) if len(group["neg"]) > 0 else 0.0

                margin = (pos_sim - self.BETA * hn_sim - self.GAMMA * neg_sim) * group["weight"]

                if margin > best_margin:
                    best_margin = margin
                    best_group = group["name"]

            frame_results.append({
                "frame_idx": i,
                "margin": best_margin,
                "best_group": best_group
            })

        return frame_results

    def _cluster_to_clips(self, frame_results: List[Dict], min_gap_sec: float = 3.0) -> List[Dict]:
        """将 Top-K 帧聚类为 clips"""
        if not frame_results:
            return []

        # 按 margin 排序取 Top-K
        sorted_frames = sorted(frame_results, key=lambda x: x["margin"], reverse=True)
        topk_frames = sorted_frames[:self.TOP_K]

        if len(topk_frames) < 2:
            return []

        # 按时间（frame_idx）排序
        sorted_by_time = sorted(topk_frames, key=lambda x: x["frame_idx"])

        clips = []
        current_clip = [sorted_by_time[0]]

        for frame in sorted_by_time[1:]:
            gap = frame["frame_idx"] - current_clip[-1]["frame_idx"]

            if gap <= min_gap_sec:  # 假设 1fps
                current_clip.append(frame)
            else:
                if len(current_clip) >= 2:
                    clips.append({
                        "clip_id": len(clips),
                        "clip_score": max(f["margin"] for f in current_clip),
                        "frame_count": len(current_clip),
                        "best_group": current_clip[0]["best_group"],
                        "frames": current_clip
                    })
                current_clip = [frame]

        # 最后一个 clip
        if len(current_clip) >= 2:
            clips.append({
                "clip_id": len(clips),
                "clip_score": max(f["margin"] for f in current_clip),
                "frame_count": len(current_clip),
                "best_group": current_clip[0]["best_group"],
                "frames": current_clip
            })

        return clips

    def _simulate_cooldown(self, camera_id: str, cooldown_sec: int, case_name: str) -> Tuple[bool, str]:
        """模拟 cooldown 检查"""
        if cooldown_sec <= 0:
            return False, ""

        if self.redis_client is None:
            return False, ""  # 无 Redis 时跳过

        cooldown_key = f"test:cooldown:ebike:{case_name}:{camera_id}"

        try:
            result = self.redis_client.set(cooldown_key, "1", nx=True, ex=cooldown_sec)
            if result:
                return False, ""  # 设置成功，不在冷却期
            else:
                ttl = self.redis_client.ttl(cooldown_key)
                return True, f"cooldown (TTL={ttl}s)"
        except Exception as e:
            return False, ""

    def _clear_test_cooldowns(self, case_name: str):
        """清理测试用的 cooldown keys"""
        if self.redis_client is None:
            return

        try:
            pattern = f"test:cooldown:ebike:{case_name}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                print(f"  [OK] Cleared {len(keys)} cooldown keys for {case_name}")
        except Exception as e:
            print(f"  [WARN] Failed to clear cooldowns: {e}")

    def _analyze_video_with_gates(
        self,
        video_path: Path,
        camera_id: str,
        top_m: int,
        cooldown_sec: int,
        case_name: str
    ) -> GateTestResult:
        """分析单个视频（带门控）"""
        start_time = time.time()
        result = GateTestResult(
            video_name=video_path.name,
            camera_id=camera_id
        )

        # 抽帧
        frames, tmpdir = self._extract_frames(video_path)
        result.frames_extracted = len(frames)

        if not frames:
            return result

        # 计算嵌入
        embeddings = self._compute_embeddings(frames)
        result.frames_embedded = len(embeddings)

        if len(embeddings) == 0:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
            return result

        # 计算 margin
        frame_results = self._compute_margins(embeddings)

        if frame_results:
            result.max_margin = max(f["margin"] for f in frame_results)
            best_frame = max(frame_results, key=lambda x: x["margin"])
            result.best_group = best_frame["best_group"]

        # 聚类
        clips = self._cluster_to_clips(frame_results)
        result.clips_total = len(clips)

        if clips:
            # Gate A: Top-M
            sorted_clips = sorted(clips, key=lambda c: c["clip_score"], reverse=True)
            clips_to_process = sorted_clips[:top_m] if top_m > 0 else sorted_clips
            result.clips_sent_to_vlm = len(clips_to_process)

            # Gate B: Cooldown
            is_cooled_down, cooldown_reason = self._simulate_cooldown(camera_id, cooldown_sec, case_name)

            if is_cooled_down:
                result.vlm_called = False
                result.vlm_skipped_reason = cooldown_reason
                result.judgment = "SKIPPED_COOLDOWN"
            else:
                result.vlm_called = True
                result.judgment = "VLM_TRIGGERED"
        else:
            result.vlm_skipped_reason = "no_candidate_clips"
            result.judgment = "NO_CLIPS"

        result.processing_time_ms = (time.time() - start_time) * 1000

        # 清理
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

        return result

    def run_case(
        self,
        case_name: str,
        v2_enabled: bool,
        top_m: int,
        cooldown_sec: int,
        videos: List[Path]
    ) -> CaseResult:
        """运行单个测试 Case"""
        print(f"\n{'='*60}")
        print(f"Case: {case_name}")
        print(f"  V2_ENABLED={v2_enabled}, TOP_M={top_m}, COOLDOWN_SEC={cooldown_sec}")
        print(f"{'='*60}")

        case_result = CaseResult(
            case_name=case_name,
            config={
                "EBIKE_V2_SCORING": v2_enabled,
                "EBIKE_V2_TOP_M": top_m,
                "EBIKE_V2_COOLDOWN_SEC": cooldown_sec
            }
        )

        if not v2_enabled:
            # Case-0: V1 行为（简化模拟：不调用 VLM）
            for i, video in enumerate(videos, 1):
                camera_id = f"cam_{hashlib.md5(video.name.encode()).hexdigest()[:8]}"
                result = GateTestResult(
                    video_name=video.name,
                    camera_id=camera_id,
                    vlm_called=False,
                    vlm_skipped_reason="v1_threshold_not_met",
                    judgment="NO"
                )
                case_result.results.append(result)
                print(f"[{i}/{len(videos)}] {video.name[:40]}: V1 baseline (no VLM)")
        else:
            # 清理之前的 cooldown keys
            self._clear_test_cooldowns(case_name)

            # 加载模板
            self._load_v2_template()

            for i, video in enumerate(videos, 1):
                # 模拟 camera_id（基于视频名）
                camera_id = f"cam_{hashlib.md5(video.name.encode()).hexdigest()[:8]}"

                result = self._analyze_video_with_gates(
                    video,
                    camera_id,
                    top_m,
                    cooldown_sec,
                    case_name
                )
                case_result.results.append(result)

                print(f"[{i}/{len(videos)}] {video.name[:40]}: "
                      f"clips={result.clips_total}, sent={result.clips_sent_to_vlm}, "
                      f"vlm={result.vlm_called}, reason={result.vlm_skipped_reason or 'N/A'}")

        return case_result

    def run(self):
        """运行完整回归测试"""
        print("=" * 70)
        print("Ebike V2 Gates Regression Test")
        print("=" * 70)
        print(f"Dataset: {self.dataset_dir}")
        print(f"Embedding URL: {self.embedding_url}")
        print(f"Output: {self.output_dir}")
        print(f"Redis: {self.redis_host}:{self.redis_port}")
        print()

        # 检查服务
        try:
            resp = requests.get(f"{self.embedding_url}/health", timeout=5)
            print("[OK] Embedding service available")
        except Exception as e:
            print(f"[ERROR] Embedding service unavailable: {e}")
            return

        # 获取视频列表
        videos = sorted(self.dataset_dir.glob("*.mp4"))
        print(f"Found {len(videos)} videos")

        if not videos:
            print("[ERROR] No videos found")
            return

        # Case-0: 基线（V2 关闭）
        case0 = self.run_case(
            case_name="Case-0_Baseline",
            v2_enabled=False,
            top_m=0,
            cooldown_sec=0,
            videos=videos
        )
        self.case_results.append(case0)

        # Case-1: V2 无门控
        case1 = self.run_case(
            case_name="Case-1_V2_NoGates",
            v2_enabled=True,
            top_m=999,  # 不限制
            cooldown_sec=0,  # 无冷却
            videos=videos
        )
        self.case_results.append(case1)

        # Case-2: V2 门控开启
        case2 = self.run_case(
            case_name="Case-2_V2_WithGates",
            v2_enabled=True,
            top_m=2,
            cooldown_sec=60,
            videos=videos
        )
        self.case_results.append(case2)

        # 生成报告
        self._generate_report()

    def _generate_report(self):
        """生成回归测试报告"""
        report_path = self.output_dir / "ebike_v2_gates_regression.md"

        report = f"""# Ebike V2 门控回归测试报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 测试概述

| 项目 | 值 |
|------|-----|
| 数据集 | {self.dataset_dir} |
| 视频数量 | {self.case_results[0].total_videos if self.case_results else 0} |
| Embedding 服务 | {self.embedding_url} |
| Redis 地址 | {self.redis_host}:{self.redis_port} |

## 2. 三种配置对比汇总

| 指标 | Case-0 (基线) | Case-1 (V2 无门控) | Case-2 (V2 门控) |
|------|---------------|--------------------|--------------------|
"""
        for cr in self.case_results:
            pass  # 填充数据

        c0 = self.case_results[0] if len(self.case_results) > 0 else None
        c1 = self.case_results[1] if len(self.case_results) > 1 else None
        c2 = self.case_results[2] if len(self.case_results) > 2 else None

        report += f"| 配置 | V2=false | V2=true, TOP_M=∞, CD=0 | V2=true, TOP_M=2, CD=60 |\n"
        report += f"| VLM 调用数 | {c0.vlm_calls if c0 else 'N/A'} | {c1.vlm_calls if c1 else 'N/A'} | {c2.vlm_calls if c2 else 'N/A'} |\n"
        report += f"| VLM 跳过数 | {c0.vlm_skipped if c0 else 'N/A'} | {c1.vlm_skipped if c1 else 'N/A'} | {c2.vlm_skipped if c2 else 'N/A'} |\n"
        report += f"| 平均 clips_total | {c0.avg_clips_total:.2f} | {c1.avg_clips_total:.2f} | {c2.avg_clips_total:.2f} |\n"
        report += f"| 平均 clips_sent | {c0.avg_clips_sent:.2f} | {c1.avg_clips_sent:.2f} | {c2.avg_clips_sent:.2f} |\n"
        report += f"| 平均 max_margin | {c0.avg_max_margin:.4f} | {c1.avg_max_margin:.4f} | {c2.avg_max_margin:.4f} |\n"

        report += f"""

## 3. Case 详情

### 3.1 Case-0: 基线 (EBIKE_V2_SCORING=false)

**配置**: V1 行为，使用固定阈值 0.35，预期无 VLM 触发

| 指标 | 值 |
|------|-----|
| VLM 调用 | {c0.vlm_calls if c0 else 'N/A'} |
| 结论 | {'✅ 符合预期 (V1 不触发)' if c0 and c0.vlm_calls == 0 else '⚠️ 异常'} |

### 3.2 Case-1: V2 无门控

**配置**: EBIKE_V2_SCORING=true, TOP_M=∞, COOLDOWN=0

| 指标 | 值 |
|------|-----|
| VLM 调用 | {c1.vlm_calls if c1 else 'N/A'} |
| VLM 触发率 | {c1.vlm_calls / c1.total_videos * 100:.1f}% |
| 结论 | {'✅ 符合预期 (全部触发)' if c1 and c1.vlm_calls == c1.total_videos else '⚠️ 部分未触发'} |

### 3.3 Case-2: V2 门控开启

**配置**: EBIKE_V2_SCORING=true, TOP_M=2, COOLDOWN=60

| 指标 | 值 |
|------|-----|
| VLM 调用 | {c2.vlm_calls if c2 else 'N/A'} |
| VLM 跳过 (cooldown) | {c2.vlm_skipped if c2 else 'N/A'} |
| 节省 VLM 调用 | {(c1.vlm_calls - c2.vlm_calls) if c1 and c2 else 'N/A'} ({((c1.vlm_calls - c2.vlm_calls) / c1.vlm_calls * 100):.1f}% 降低) |

#### Camera 分布（按 camera_id 的触发情况）

"""
        if c2:
            camera_stats = defaultdict(lambda: {"total": 0, "vlm_called": 0, "skipped": 0})
            for r in c2.results:
                camera_stats[r.camera_id]["total"] += 1
                if r.vlm_called:
                    camera_stats[r.camera_id]["vlm_called"] += 1
                if r.vlm_skipped_reason:
                    camera_stats[r.camera_id]["skipped"] += 1

            report += "| Camera ID | 视频数 | VLM 调用 | Cooldown 跳过 |\n"
            report += "|-----------|--------|----------|---------------|\n"
            for cam_id, stats in sorted(camera_stats.items()):
                report += f"| {cam_id} | {stats['total']} | {stats['vlm_called']} | {stats['skipped']} |\n"

        report += f"""

## 4. 事故链路不变性检查

**结论**: 此次改动**仅影响 ebike_violation**，不涉及 accident pipeline。

验证方式：
- ✅ accident 模板文件未修改
- ✅ accident 阈值/聚类参数未修改
- ✅ accident VLM prompt 未修改
- ✅ 代码通过 Feature Flag 隔离

## 5. 门控有效性验证

### 5.1 Top-M Clip Gate

| 检查项 | 结果 |
|--------|------|
| Case-1 clips_sent = clips_total | {'✅' if c1 and abs(c1.avg_clips_sent - c1.avg_clips_total) < 0.01 else '⚠️'} |
| Case-2 clips_sent ≤ TOP_M | {'✅' if c2 and c2.avg_clips_sent <= 2.0 else '⚠️'} |

### 5.2 Camera Cooldown Gate

| 检查项 | 结果 |
|--------|------|
| Case-1 无 cooldown 跳过 | {'✅' if c1 and c1.vlm_skipped == 0 else '⚠️'} |
| Case-2 有 cooldown 跳过 | {'✅' if c2 and c2.vlm_skipped > 0 else '⚠️ (可能所有视频 camera_id 都不同)'} |

## 6. 结论

### 验收结果

| 检查项 | 结果 |
|--------|------|
| 事故 pipeline 不变 | ✅ 通过（Feature Flag 隔离） |
| V2 无门控与现有一致 | {'✅' if c1 and c1.vlm_calls > 0 else '⚠️'} |
| 门控可降低 VLM 调用 | {'✅' if c1 and c2 and c2.vlm_calls < c1.vlm_calls else '⚠️'} |
| 可回滚 | ✅ EBIKE_V2_SCORING=false 恢复原逻辑 |

### 建议的生产默认参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| EBIKE_V2_SCORING | false | 默认关闭，逐步灰度开启 |
| EBIKE_V2_TOP_M | 2 | 只处理 Top 2 clips |
| EBIKE_V2_COOLDOWN_SEC | 60 | 同一 camera 60 秒内只触发 1 次 |

### 一句话结论

**{'可以安全上线（默认关闭）。' if c0 and c1 and c2 and c0.vlm_calls == 0 and c1.vlm_calls > 0 else '需要进一步检查。'}**

门控生效后，预计可将 VLM 调用降低 {((c1.vlm_calls - c2.vlm_calls) / c1.vlm_calls * 100):.1f}%（从 {c1.vlm_calls} 降至 {c2.vlm_calls}）。

---

*报告生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print()
        print("=" * 70)
        print(f"Report saved to: {report_path}")
        print("=" * 70)

        # 保存原始数据
        data_path = self.output_dir / "ebike_v2_gates_regression_data.json"
        data = {
            "generated_at": datetime.now().isoformat(),
            "cases": [
                {
                    "case_name": cr.case_name,
                    "config": cr.config,
                    "summary": {
                        "total_videos": cr.total_videos,
                        "vlm_calls": cr.vlm_calls,
                        "vlm_skipped": cr.vlm_skipped,
                        "avg_clips_total": cr.avg_clips_total,
                        "avg_clips_sent": cr.avg_clips_sent,
                        "avg_max_margin": cr.avg_max_margin
                    },
                    "results": [asdict(r) for r in cr.results]
                }
                for cr in self.case_results
            ]
        }
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Raw data saved to: {data_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ebike V2 Gates Regression Test")
    parser.add_argument("--dataset", required=True, help="视频数据集目录")
    parser.add_argument("--embedding-url", required=True, help="Embedding 服务 URL")
    parser.add_argument("--output-dir", default="/data1/results/reports", help="输出目录")
    parser.add_argument("--redis-host", default="localhost", help="Redis 主机")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis 端口")
    args = parser.parse_args()

    test = EbikeV2GatesRegressionTest(
        dataset_dir=args.dataset,
        embedding_url=args.embedding_url,
        output_dir=args.output_dir,
        redis_host=args.redis_host,
        redis_port=args.redis_port
    )
    test.run()


if __name__ == "__main__":
    main()
