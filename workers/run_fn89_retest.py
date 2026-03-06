#!/usr/bin/env python3
"""
重新测试 89 个 FN 视频，验证 clip_score 公式修复效果

使用方法:
    python workers/run_fn89_retest.py --output-dir /data1/eval_results/fn89_retest_v12

验证目标:
    - 30 个 clip_score_low 视频的 clip_score 是否提升
    - VLM 调用率是否提高
    - 整体 Recall 改善
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import redis

# ========== 配置 ==========
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "/data1/results")

# 数据集路径映射
DATASET_PATHS = {
    "大样本事故数据集": "/data1/videos/vaildata/大样本事故数据集",
    "大样本非交通事故数据集": "/data1/videos/vaildata/大样本非交通事故数据集",
    "FN27_2026.2.4": "/data1/videos/vaildata/FN27_2026.2.4",
    "FN400": "/data1/videos/vaildata/FN400",
    "NO_CLIP_PASS": "/data1/videos/vaildata/NO_CLIP_PASS",
}

# 89 个 FN 视频列表 (从评测报告提取)
FN_VIDEOS = {
    # Clip 分数低于阈值 (30个) - clip_score 修复的主要验证目标
    "clip_score_low": [
        ("大样本事故数据集", "211.mp4"),
        ("大样本事故数据集", "248.mp4"),
        ("大样本事故数据集", "146.mp4"),
        ("大样本事故数据集", "226.mp4"),
        ("大样本事故数据集", "175.mp4"),
        ("大样本事故数据集", "141.mp4"),
        ("大样本事故数据集", "165.mp4"),
        ("大样本事故数据集", "196.mp4"),
        ("大样本事故数据集", "261.mp4"),
        ("大样本事故数据集", "185.mp4"),
        ("大样本事故数据集", "231.mp4"),
        ("大样本事故数据集", "283.mp4"),
        ("大样本事故数据集", "186.mp4"),
        ("大样本事故数据集", "247.mp4"),
        ("FN27_2026.2.4", "211.mp4"),
        ("FN27_2026.2.4", "118.mp4"),
        ("FN27_2026.2.4", "226.mp4"),
        ("FN27_2026.2.4", "127.mp4"),
        ("FN27_2026.2.4", "196.mp4"),
        ("FN27_2026.2.4", "117.mp4"),
        ("FN27_2026.2.4", "160.mp4"),
        ("FN27_2026.2.4", "261.mp4"),
        ("FN27_2026.2.4", "247.mp4"),
        ("FN27_2026.2.4", "231.mp4"),
        ("FN27_2026.2.4", "283.mp4"),
        ("FN27_2026.2.4", "82.mp4"),
        ("NO_CLIP_PASS", "118.mp4"),
        ("NO_CLIP_PASS", "196.mp4"),
        ("NO_CLIP_PASS", "231.mp4"),
        ("NO_CLIP_PASS", "78.mp4"),
    ],

    # VLM 判定 NO (41个)
    "vlm_no": [
        ("大样本事故数据集", "101.mp4"),
        ("大样本事故数据集", "116.mp4"),
        ("大样本事故数据集", "126.mp4"),
        ("大样本事故数据集", "151.mp4"),
        ("大样本事故数据集", "152.mp4"),
        ("大样本事故数据集", "153.mp4"),
        ("大样本事故数据集", "156.mp4"),
        ("大样本事故数据集", "177.mp4"),
        ("大样本事故数据集", "182.mp4"),
        ("大样本事故数据集", "183.mp4"),
        ("大样本事故数据集", "193.mp4"),
        ("大样本事故数据集", "202.mp4"),
        ("大样本事故数据集", "203.mp4"),
        ("大样本事故数据集", "221.mp4"),
        ("大样本事故数据集", "229.mp4"),
        ("大样本事故数据集", "270.mp4"),
        ("大样本事故数据集", "82.mp4"),
        ("大样本事故数据集", "83.mp4"),
        ("大样本事故数据集", "96.mp4"),
        ("FN27_2026.2.4", "105.mp4"),
        ("FN27_2026.2.4", "119.mp4"),
        ("FN27_2026.2.4", "125.mp4"),
        ("FN27_2026.2.4", "145.mp4"),
        ("FN27_2026.2.4", "156.mp4"),
        ("FN27_2026.2.4", "215.mp4"),
        ("FN27_2026.2.4", "229.mp4"),
        ("FN27_2026.2.4", "248.mp4"),
        ("FN27_2026.2.4", "258.mp4"),
        ("FN27_2026.2.4", "270.mp4"),
        ("FN400", "198.mp4"),
        ("FN400", "221.mp4"),
        ("FN400", "271.mp4"),
        ("NO_CLIP_PASS", "105.mp4"),
        ("NO_CLIP_PASS", "128.mp4"),
        ("NO_CLIP_PASS", "156.mp4"),
        ("NO_CLIP_PASS", "160.mp4"),
        ("NO_CLIP_PASS", "181.mp4"),
        ("NO_CLIP_PASS", "193.mp4"),
        ("NO_CLIP_PASS", "229.mp4"),
        ("NO_CLIP_PASS", "258.mp4"),
        ("NO_CLIP_PASS", "82.mp4"),
    ],

    # VLM 判定 UNCERTAIN (11个)
    "vlm_uncertain": [
        ("大样本事故数据集", "102.mp4"),
        ("大样本事故数据集", "145.mp4"),
        ("大样本事故数据集", "174.mp4"),
        ("大样本事故数据集", "190.mp4"),
        ("大样本事故数据集", "201.mp4"),
        ("大样本事故数据集", "215.mp4"),
        ("大样本事故数据集", "223.mp4"),
        ("大样本事故数据集", "258.mp4"),
        ("大样本事故数据集", "78.mp4"),
        ("大样本事故数据集", "81.mp4"),
        ("NO_CLIP_PASS", "284.mp4"),
    ],

    # 无结果文件 (6个)
    "no_result": [
        ("大样本事故数据集", "460-205兴贸北街与融商五路交叉口南向北电警1_20251230181026046.mp4"),
        ("大样本事故数据集", "事故2高清.mp4"),
        ("FN27_2026.2.4", "460-202兴贸北街与融商五路交叉口东向西电警2_20251230180438758.mp4"),
        ("FN27_2026.2.4", "460-205兴贸北街与融商五路交叉口南向北电警1_20251230181026046.mp4"),
        ("FN400", "214.mp4"),
        ("FN400", "460-205兴贸北街与融商五路交叉口南向北电警1_20251230181026046.mp4"),
    ],

    # VLM 连接错误 (1个)
    "vlm_error": [
        ("大样本事故数据集", "135.mp4"),
    ],
}


class FN89Retester:
    """89 个 FN 视频重测试器"""

    def __init__(self, output_dir: str, batch_size: int = 8, timeout: int = 120):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.timeout = timeout  # 每批次超时时间 (秒)

        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.results: Dict[str, Dict] = {}
        self.start_time = datetime.now()

    def get_video_path(self, dataset: str, filename: str) -> Optional[str]:
        """获取视频完整路径"""
        base_path = DATASET_PATHS.get(dataset)
        if not base_path:
            print(f"[WARN] 未知数据集: {dataset}")
            return None

        video_path = Path(base_path) / filename
        return str(video_path)

    def inject_task(self, job_id: str, video_path: str, camera_id: str) -> bool:
        """注入单个任务到 video_tasks 队列"""
        try:
            import uuid
            trace_id = str(uuid.uuid4())[:8]
            task_data = {
                "job_id": job_id,
                "camera_id": camera_id,
                "window_path": video_path,
                "analysis_type": "accident",
                "trace_id": trace_id,
                "injected_at": datetime.now().isoformat(),
            }
            self.redis.xadd("video_tasks", task_data)
            return True
        except Exception as e:
            print(f"[ERROR] 注入任务失败: {job_id} - {e}")
            return False

    def wait_for_results(self, job_ids: List[str], timeout: int) -> Dict[str, Dict]:
        """等待批次结果"""
        results = {}
        start = time.time()
        pending = set(job_ids)

        while pending and (time.time() - start) < timeout:
            for job_id in list(pending):
                # 检查结果文件
                result_path = Path(RESULTS_DIR) / "accident" / job_id / "result.json"
                if result_path.exists():
                    try:
                        with open(result_path) as f:
                            results[job_id] = json.load(f)
                        pending.remove(job_id)
                        print(f"  [OK] {job_id} - 结果已生成")
                    except Exception as e:
                        print(f"  [WARN] {job_id} - 读取结果失败: {e}")

            if pending:
                time.sleep(2)

        # 标记超时的任务
        for job_id in pending:
            results[job_id] = {"error": "timeout", "status": "timeout"}
            print(f"  [TIMEOUT] {job_id}")

        return results

    def run_batch(self, videos: List[Tuple[str, str]], batch_name: str, start_idx: int) -> Dict[str, Dict]:
        """运行单个批次"""
        batch_results = {}
        batch_jobs = []

        print(f"\n{'='*60}")
        print(f"批次: {batch_name} ({len(videos)} 个视频)")
        print(f"{'='*60}")

        # 注入任务
        for i, (dataset, filename) in enumerate(videos):
            job_id = f"fn89_{batch_name}_{start_idx + i:04d}"
            video_path = self.get_video_path(dataset, filename)

            if not video_path:
                batch_results[job_id] = {"error": "invalid_path", "dataset": dataset, "filename": filename}
                continue

            camera_id = f"fn89_{batch_name}_{filename}"
            if self.inject_task(job_id, video_path, camera_id):
                batch_jobs.append(job_id)
                print(f"  [INJECT] {job_id}: {dataset}/{filename}")

        # 等待结果
        if batch_jobs:
            print(f"\n等待 {len(batch_jobs)} 个任务完成 (超时: {self.timeout}s)...")
            results = self.wait_for_results(batch_jobs, self.timeout)
            batch_results.update(results)

        return batch_results

    def analyze_results(self, category: str, videos: List[Tuple[str, str]], results: Dict[str, Dict]) -> Dict:
        """分析单个类别的结果"""
        analysis = {
            "category": category,
            "total": len(videos),
            "success": 0,
            "vlm_called": 0,
            "vlm_yes": 0,
            "vlm_no": 0,
            "vlm_uncertain": 0,
            "clip_score_improved": 0,
            "timeout": 0,
            "error": 0,
            "details": [],
        }

        for job_id, result in results.items():
            if "error" in result:
                if result.get("status") == "timeout":
                    analysis["timeout"] += 1
                else:
                    analysis["error"] += 1
                continue

            analysis["success"] += 1

            # 提取关键指标
            vlm_called = result.get("vlm_called", False)
            decision = result.get("decision", "")
            clip_score = result.get("max_clip_score", 0)

            if vlm_called:
                analysis["vlm_called"] += 1
                if decision == "YES":
                    analysis["vlm_yes"] += 1
                elif decision == "NO":
                    analysis["vlm_no"] += 1
                elif decision == "UNCERTAIN":
                    analysis["vlm_uncertain"] += 1

            # clip_score 提升判断 (原来 < 0.10)
            if clip_score >= 0.10:
                analysis["clip_score_improved"] += 1

            analysis["details"].append({
                "job_id": job_id,
                "clip_score": clip_score,
                "vlm_called": vlm_called,
                "decision": decision,
            })

        return analysis

    def run(self, categories: Optional[List[str]] = None):
        """运行完整测试"""
        if categories is None:
            categories = ["clip_score_low"]  # 默认只测试 clip_score_low

        all_results = {}
        all_analysis = {}

        for category in categories:
            if category not in FN_VIDEOS:
                print(f"[WARN] 未知类别: {category}")
                continue

            videos = FN_VIDEOS[category]
            category_results = {}

            # 分批处理
            for i in range(0, len(videos), self.batch_size):
                batch = videos[i:i + self.batch_size]
                batch_results = self.run_batch(batch, category, i)
                category_results.update(batch_results)

                # 批次间等待
                if i + self.batch_size < len(videos):
                    print("\n等待 5 秒后继续下一批次...")
                    time.sleep(5)

            all_results[category] = category_results
            all_analysis[category] = self.analyze_results(category, videos, category_results)

        # 保存结果
        self.save_results(all_results, all_analysis)

        # 打印摘要
        self.print_summary(all_analysis)

    def save_results(self, results: Dict, analysis: Dict):
        """保存结果到文件"""
        # 原始结果
        with open(self.output_dir / "raw_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 分析结果
        with open(self.output_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        # 汇总指标
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "duration_sec": (datetime.now() - self.start_time).total_seconds(),
            "categories": {},
        }

        for cat, data in analysis.items():
            metrics["categories"][cat] = {
                "total": data["total"],
                "success": data["success"],
                "vlm_called": data["vlm_called"],
                "vlm_call_rate": data["vlm_called"] / data["success"] if data["success"] > 0 else 0,
                "vlm_yes": data["vlm_yes"],
                "clip_score_improved": data["clip_score_improved"],
                "improvement_rate": data["clip_score_improved"] / data["total"] if data["total"] > 0 else 0,
            }

        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存到: {self.output_dir}")

    def print_summary(self, analysis: Dict):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("测试摘要")
        print("="*60)

        for cat, data in analysis.items():
            print(f"\n【{cat}】({data['total']} 个视频)")
            print(f"  成功: {data['success']}")
            print(f"  VLM 调用: {data['vlm_called']} ({data['vlm_called']/data['success']*100:.1f}%)" if data['success'] > 0 else "  VLM 调用: 0")
            print(f"  VLM 判定 YES: {data['vlm_yes']}")
            print(f"  VLM 判定 NO: {data['vlm_no']}")
            print(f"  VLM 判定 UNCERTAIN: {data['vlm_uncertain']}")
            print(f"  clip_score >= 0.10: {data['clip_score_improved']} ({data['clip_score_improved']/data['total']*100:.1f}%)")
            print(f"  超时: {data['timeout']}")
            print(f"  错误: {data['error']}")


def main():
    parser = argparse.ArgumentParser(description="重新测试 89 个 FN 视频")
    parser.add_argument("--output-dir", default="/data1/eval_results/fn89_retest_v12",
                        help="输出目录")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="每批次视频数量")
    parser.add_argument("--timeout", type=int, default=120,
                        help="每批次超时时间 (秒)")
    parser.add_argument("--categories", nargs="+",
                        default=["clip_score_low"],
                        choices=["clip_score_low", "vlm_no", "vlm_uncertain", "no_result", "vlm_error", "all"],
                        help="要测试的类别")
    args = parser.parse_args()

    # 处理 all 参数
    if "all" in args.categories:
        categories = ["clip_score_low", "vlm_no", "vlm_uncertain", "no_result", "vlm_error"]
    else:
        categories = args.categories

    print(f"FN89 重测试")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"超时时间: {args.timeout}s")
    print(f"测试类别: {categories}")

    tester = FN89Retester(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    tester.run(categories=categories)


if __name__ == "__main__":
    main()
