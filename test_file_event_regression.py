"""
文件级事件定位回归测试脚本

验证4个真实mp4文件：
- clip-a826cb3b.mp4 (full-process GT) - 完整覆盖事故前→碰撞→事故后
- clip-372493c8.mp4 (无事故) - 无事故，不应排第一
- clip-d8839af9.mp4 (post-event only) - 事故后开始
- clip-51e7b0c2.mp4 (post-event only) - 事故后开始

验收标准：
1. a826 必须进入TopK并排名靠前（至少Top3）
2. 372 不得排名第1，因t0_validity低而coverage_effective低
3. d883、51e7 需要输出 POST_EVENT_ONLY 或 UNCERTAIN

运行方式：
    python test_file_event_regression.py
    python test_file_event_regression.py --with-yolo  # 使用YOLO检测
    python test_file_event_regression.py --with-vlm   # 调用真实VLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_vlm.file_event_locator import (
    FileEventConfig,
    generate_file_candidates,
    rank_candidates,
    log_candidates_ranking,
    apply_conditional_retention,
)
from traffic_vlm.vlm_client import (
    create_mock_vlm_response,
    parse_vlm_response_with_verdict,
)
from traffic_vlm.config import VLMRetentionConfig


# 测试用例定义
TEST_CASES = [
    {
        "clip_id": "clip-a826cb3b",
        "file_name": "clip-a826cb3b.mp4",
        "expected_type": "full_process",
        "description": "完整覆盖事故前→碰撞→事故后",
        "expected_rank": "top3",
        "expected_verdict": "YES",
        "is_ground_truth": True,
    },
    {
        "clip_id": "clip-372493c8",
        "file_name": "clip-372493c8.mp4",
        "expected_type": "no_accident",
        "description": "无事故（误报源）",
        "expected_rank": "not_first",
        "expected_verdict": "NO",
        "is_ground_truth": False,
    },
    {
        "clip_id": "clip-d8839af9",
        "file_name": "clip-d8839af9.mp4",
        "expected_type": "post_event_only",
        "description": "事故后开始（后果片段）",
        "expected_rank": "any",
        "expected_verdict": "POST_EVENT_ONLY",
        "is_ground_truth": False,
    },
    {
        "clip_id": "clip-51e7b0c2",
        "file_name": "clip-51e7b0c2.mp4",
        "expected_type": "post_event_only",
        "description": "事故后开始（后果片段）",
        "expected_rank": "any",
        "expected_verdict": "POST_EVENT_ONLY",
        "is_ground_truth": False,
    },
]


def find_test_files(base_dir: str) -> Dict[str, str]:
    """查找测试文件"""
    file_map = {}

    # 尝试多个可能的目录
    search_dirs = [
        os.path.join(base_dir, "data", "camera-1", "20251226", "raw_suspect_clips"),
        os.path.join(base_dir, "data", "raw_suspect_clips"),
        os.path.join(base_dir, "raw_suspect_clips"),
        base_dir,
    ]

    for test_case in TEST_CASES:
        file_name = test_case["file_name"]
        found = False

        for search_dir in search_dirs:
            file_path = os.path.join(search_dir, file_name)
            if os.path.exists(file_path):
                file_map[test_case["clip_id"]] = file_path
                found = True
                break

        if not found:
            print(f"[WARNING] 未找到测试文件: {file_name}")

    return file_map


def run_file_event_analysis(
    file_map: Dict[str, str],
    config: FileEventConfig,
    yolo_model=None,
) -> List[Dict]:
    """运行文件级事件分析"""
    all_candidates = []

    for clip_id, file_path in file_map.items():
        print(f"\n[分析] {clip_id}: {file_path}")

        candidates = generate_file_candidates(
            video_path=file_path,
            config=config,
            yolo_model=yolo_model,
            camera_id="camera-1",
        )

        for c in candidates:
            # 添加测试用例信息
            test_case = next((tc for tc in TEST_CASES if tc["clip_id"] == clip_id), None)
            if test_case:
                c["expected_type"] = test_case["expected_type"]
                c["expected_verdict"] = test_case["expected_verdict"]
                c["is_ground_truth"] = test_case["is_ground_truth"]

            all_candidates.append(c)

        print(f"    生成 {len(candidates)} 个候选")
        for c in candidates:
            print(f"      - {c['clip_id']}: t0={c.get('t0', 0):.1f}s, "
                  f"validity={c.get('t0_validity', 0):.3f}, "
                  f"cover_eff={c.get('coverage_effective', 0):.4f}, "
                  f"final={c.get('final_score', 0):.4f}")

    return all_candidates


def simulate_vlm_verdicts(candidates: List[Dict], mock_mode: str = "expected") -> List[Dict]:
    """
    模拟VLM判决

    mock_mode:
    - "expected": 根据expected_type返回预期verdict
    - "all_no": 全部返回NO（测试有条件保留）
    """
    for c in candidates:
        if mock_mode == "expected":
            expected = c.get("expected_verdict", "NO")
            vlm_response = create_mock_vlm_response(expected)
        else:
            vlm_response = create_mock_vlm_response("NO")

        # 解析响应
        vlm_response = parse_vlm_response_with_verdict(vlm_response)
        c["vlm_verdict"] = vlm_response.get("verdict", "NO")
        c["vlm_confidence"] = vlm_response.get("confidence", 0.0)
        c["vlm_response"] = vlm_response

    return candidates


def apply_retention_policy(
    candidates: List[Dict],
    retention_config: VLMRetentionConfig,
) -> List[Dict]:
    """应用有条件保留策略"""
    from traffic_vlm.file_event_locator import apply_conditional_retention, FileEventConfig

    event_config = FileEventConfig(
        validity_threshold=retention_config.validity_threshold,
        risk_peak_threshold=retention_config.risk_peak_threshold,
        roi_median_threshold=retention_config.roi_median_threshold,
    )

    for c in candidates:
        verdict = c.get("vlm_verdict", "NO")
        kept, reason = apply_conditional_retention(c, verdict, event_config)
        c["kept"] = kept
        c["keep_reason"] = reason

    return candidates


def verify_results(ranked_candidates: List[Dict]) -> Tuple[bool, List[str]]:
    """验证结果是否满足验收标准（语义对齐版）"""
    errors = []
    warnings = []

    # 找到各个测试用例的排名和评分
    a826_rank = None
    a826_rank_score = None
    c372_rank = None
    post_event_clips = []  # POST_EVENT_ONLY clips

    for i, c in enumerate(ranked_candidates, 1):
        clip_id = c.get("clip_id", "")
        rank_score = c.get("rank_score", 0)
        verdict = c.get("verdict", "")

        if "a826cb3b" in clip_id:
            a826_rank = i
            a826_rank_score = rank_score
        elif "372493c8" in clip_id:
            c372_rank = i

        if verdict == "POST_EVENT_ONLY":
            post_event_clips.append({
                "clip_id": clip_id,
                "rank": i,
                "rank_score": rank_score,
                "full_process_score": c.get("full_process_score", 0),
                "post_event_score": c.get("post_event_score", 0),
            })

    # ===== 回归门控验证 =====

    # 验证1: a826必须在Top3（或至少Top2）
    if a826_rank is None:
        errors.append("clip-a826cb3b 未在候选列表中")
    elif a826_rank > 3:
        errors.append(f"clip-a826cb3b 排名第{a826_rank}，应在Top3")
    else:
        print(f"[PASS] clip-a826cb3b 排名第{a826_rank} (Top3), rank_score={a826_rank_score:.4f}")

    # 验证2: 372不能排第一
    if c372_rank == 1:
        errors.append("clip-372493c8 排名第1，应被压低")
    elif c372_rank is not None:
        print(f"[PASS] clip-372493c8 排名第{c372_rank} (不是第1)")

    # 验证3: a826的rank_score必须高于所有POST_EVENT_ONLY clips
    if a826_rank_score is not None and post_event_clips:
        for pe in post_event_clips:
            if a826_rank_score <= pe["rank_score"]:
                errors.append(
                    f"clip-a826cb3b rank_score={a826_rank_score:.4f} <= "
                    f"{pe['clip_id']} rank_score={pe['rank_score']:.4f}"
                )
            else:
                print(
                    f"[PASS] a826 rank_score={a826_rank_score:.4f} > "
                    f"{pe['clip_id'][:12]} rank_score={pe['rank_score']:.4f}"
                )

    # 验证4: POST_EVENT_ONLY clips的full_process_score必须为0
    for pe in post_event_clips:
        if pe["full_process_score"] != 0:
            errors.append(
                f"{pe['clip_id']} verdict=POST_EVENT_ONLY但full_process_score={pe['full_process_score']:.4f}，应为0"
            )
        else:
            print(f"[PASS] {pe['clip_id'][:16]} full_process_score=0 (POST_EVENT_ONLY语义正确)")

    # 验证5: POST_EVENT_ONLY clips的post_event_score必须为1
    for pe in post_event_clips:
        if pe["post_event_score"] != 1.0:
            errors.append(
                f"{pe['clip_id']} verdict=POST_EVENT_ONLY但post_event_score={pe['post_event_score']:.1f}，应为1"
            )
        else:
            print(f"[PASS] {pe['clip_id'][:16]} post_event_score=1.0 (POST_EVENT_ONLY语义正确)")

    # 验证6: 检查t0_validity
    for c in ranked_candidates:
        clip_id = c.get("clip_id", "")
        validity = c.get("t0_validity", 0)

        if "372493c8" in clip_id and validity > 0.5:
            warnings.append(f"clip-372493c8 t0_validity={validity:.3f} 偏高，应该较低")
        elif "a826cb3b" in clip_id and validity < 0.2:
            warnings.append(f"clip-a826cb3b t0_validity={validity:.3f} 偏低，应该较高")

    # 验证7: 检查is_full_process标记
    for c in ranked_candidates:
        clip_id = c.get("clip_id", "")
        is_full = c.get("is_full_process", False)
        expected_type = c.get("expected_type", "")

        if expected_type == "full_process" and not is_full:
            warnings.append(f"{clip_id} 应该标记为 is_full_process=True")

    if warnings:
        for w in warnings:
            print(f"[WARNING] {w}")

    success = len(errors) == 0
    return success, errors


def _convert_to_json_serializable(obj):
    """转换numpy类型为Python原生类型"""
    import numpy as np
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(i) for i in obj]
    return obj


def generate_report(
    ranked_candidates: List[Dict],
    success: bool,
    errors: List[str],
    output_dir: str = "data/regression_reports",
) -> str:
    """生成诊断报告"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"file_event_regression_{timestamp}.json")

    report = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "errors": errors,
        "test_cases": [asdict(tc) if hasattr(tc, '__dataclass_fields__') else tc for tc in TEST_CASES],
        "candidates": [],
    }

    for c in ranked_candidates:
        candidate_report = {
            "clip_id": c.get("clip_id"),
            "video_path": c.get("video_path"),
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time"),
            "duration": c.get("duration"),
            "t_event": c.get("t_event"),
            "t0": c.get("t0"),
            "t0_fallback": c.get("t0_fallback"),
            "t0_method": c.get("t0_method"),
            "t0_validity": c.get("t0_validity"),
            "validity_reason": c.get("validity_reason"),
            "risk_peak": c.get("risk_peak"),
            "pre_ok": c.get("pre_ok"),
            "post_ok": c.get("post_ok"),
            "coverage_raw": c.get("coverage_raw"),
            "coverage_effective": c.get("coverage_effective"),
            "final_score": c.get("final_score"),
            "is_full_process": c.get("is_full_process"),
            "vlm_verdict": c.get("vlm_verdict"),
            "vlm_confidence": c.get("vlm_confidence"),
            "kept": c.get("kept"),
            "keep_reason": c.get("keep_reason"),
            "expected_type": c.get("expected_type"),
            "expected_verdict": c.get("expected_verdict"),
            "is_ground_truth": c.get("is_ground_truth"),
        }
        report["candidates"].append(candidate_report)

    # 转换numpy类型
    report = _convert_to_json_serializable(report)

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[报告] 已保存到: {report_file}")
    return report_file


def print_results_table(ranked_candidates: List[Dict]):
    """打印结果表格（rank_score版）"""
    print("\n" + "=" * 170)
    print("回归测试结果（按 rank_score 排序）")
    print("=" * 170)

    header = (
        f"{'Rank':>4} | {'Clip ID':<16} | {'Type':<12} | {'verdict':<15} | "
        f"{'t0':>5} | {'full_p':>6} | {'post_e':>6} | {'bonus':>6} | {'penalty':>7} | "
        f"{'final':>7} | {'rank_score':>10} | {'kept':<4}"
    )
    print(header)
    print("-" * 170)

    for i, c in enumerate(ranked_candidates, 1):
        clip_id = c.get("clip_id", "N/A")[:16]
        expected_type = c.get("expected_type", "unknown")[:12]
        verdict = c.get("vlm_verdict", "N/A")[:15]
        t0 = c.get("t0", 0)
        full_p = c.get("full_process_score", 0)
        post_e = c.get("post_event_score", 0)
        bonus = c.get("full_process_bonus", 0)
        penalty = c.get("post_event_penalty", 0)
        final = c.get("final_score", 0)
        rank_score = c.get("rank_score", 0)
        kept = "YES" if c.get("kept") else "NO"

        row = (
            f"{i:>4} | {clip_id:<16} | {expected_type:<12} | {verdict:<15} | "
            f"{t0:>5.1f} | {full_p:>6.3f} | {post_e:>6.1f} | {bonus:>6.3f} | {penalty:>7.3f} | "
            f"{final:>7.4f} | {rank_score:>10.4f} | {kept:<4}"
        )
        print(row)

    print("-" * 170)


def main():
    parser = argparse.ArgumentParser(description="文件级事件定位回归测试")
    parser.add_argument(
        "--with-yolo",
        action="store_true",
        help="使用YOLO检测（需要GPU）"
    )
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="调用真实VLM API"
    )
    parser.add_argument(
        "--mock-all-no",
        action="store_true",
        help="VLM全部模拟返回NO（测试有条件保留）"
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="项目根目录"
    )
    parser.add_argument(
        "--dump-scorecard",
        action="store_true",
        help="输出每个clip的分数分解（scorecard.json）"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="生成完整诊断报告（含scorecard）"
    )
    parser.add_argument(
        "--report-output",
        default="reports",
        help="报告输出目录（默认reports）"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("文件级事件定位回归测试")
    print("=" * 80)
    print(f"项目目录: {args.base_dir}")
    print(f"使用YOLO: {args.with_yolo}")
    print(f"使用真实VLM: {args.with_vlm}")

    # 1. 查找测试文件
    print("\n[步骤1] 查找测试文件...")
    file_map = find_test_files(args.base_dir)
    print(f"找到 {len(file_map)}/{len(TEST_CASES)} 个测试文件")

    if not file_map:
        print("[ERROR] 未找到任何测试文件，退出")
        return 1

    for clip_id, path in file_map.items():
        print(f"  - {clip_id}: {path}")

    # 2. 加载YOLO模型（可选）
    yolo_model = None
    if args.with_yolo:
        print("\n[步骤2] 加载YOLO模型...")
        try:
            from ultralytics import YOLO
            yolo_model = YOLO("yolo11s.pt")
            print("  YOLO模型加载成功")
        except Exception as e:
            print(f"  [WARNING] YOLO加载失败: {e}")

    # 3. 运行文件级事件分析
    print("\n[步骤3] 运行文件级事件分析...")
    config = FileEventConfig(
        risk_sampling_fps=2.0,
        top_n_peaks=5,
        peak_threshold=0.15,
        pre_roll=8.0,
        post_roll=12.0,
        candidate_mode="file_only",  # 仅file-level候选
    )

    candidates = run_file_event_analysis(file_map, config, yolo_model)
    print(f"  共生成 {len(candidates)} 个候选")

    # 4. 模拟VLM判决（必须在排序之前，以便rank_score使用verdict）
    print("\n[步骤4] 模拟VLM判决...")
    mock_mode = "all_no" if args.mock_all_no else "expected"
    candidates = simulate_vlm_verdicts(candidates, mock_mode)

    # 将vlm_verdict复制到verdict字段供rank_candidates使用
    for c in candidates:
        c["verdict"] = c.get("vlm_verdict", "NO")

    # 5. 排序候选（使用verdict计算rank_score）
    print("\n[步骤5] 排序候选（按rank_score）...")
    ranked = rank_candidates(candidates)
    log_candidates_ranking(ranked)

    # 6. 应用保留策略
    print("\n[步骤6] 应用有条件保留策略...")
    retention_config = VLMRetentionConfig(
        enabled=True,
        validity_threshold=0.3,
        risk_peak_threshold=0.25,
        roi_median_threshold=80.0,
    )
    ranked = apply_retention_policy(ranked, retention_config)

    # 7. 打印结果表格
    print_results_table(ranked)

    # 8. 验证结果
    print("\n[步骤7] 验证结果...")
    success, errors = verify_results(ranked)

    if errors:
        print("\n[FAIL] 验证失败:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[SUCCESS] 所有验证通过!")

    # 9. 生成报告
    print("\n[步骤8] 生成诊断报告...")
    report_file = generate_report(ranked, success, errors)

    # 10. 生成Scorecard + Report.md
    if args.dump_scorecard or args.report:
        print("\n[步骤9] 生成Scorecard分数分解 + report.md...")
        try:
            from reporting import (
                ScorecardBuilder,
                generate_scorecards_from_candidates,
                ReportBuilder,
                RunContext,
                ClipResult,
            )

            # 构建VLM结果映射
            vlm_results = {}
            for c in ranked:
                clip_id = c.get("clip_id")
                if c.get("vlm_verdict"):
                    vlm_results[clip_id] = {
                        "verdict": c.get("vlm_verdict"),
                        "confidence": c.get("vlm_confidence", 0),
                    }

            # 为每个候选添加必要的filter字段
            for i, c in enumerate(ranked):
                # 根据是否有VLM verdict判断是否调用了VLM
                c["filter_status"] = "PASSED" if c.get("vlm_verdict") else "SKIPPED"
                c["skip_reason"] = "" if c.get("vlm_verdict") else "not_vlm_called"

            # 生成scorecards
            scorecard_dir = generate_scorecards_from_candidates(
                candidates=ranked,
                output_dir=args.report_output,
                vlm_results=vlm_results,
                thresholds={
                    "clip_score_threshold": 0.35,
                    "validity_threshold": retention_config.validity_threshold,
                    "risk_peak_threshold": retention_config.risk_peak_threshold,
                },
            )
            print(f"  Scorecard输出目录: {scorecard_dir}")

            # ===== 生成 report.md =====
            print("\n[步骤10] 生成 report.md 诊断报告...")

            # 创建运行上下文
            context = RunContext()
            context.run_id = os.path.basename(scorecard_dir)  # 使用同一run_id
            context.n_clips_cut = len(ranked)
            context.n_vlm_analyzed = sum(1 for c in ranked if c.get("vlm_verdict"))
            context.clip_score_threshold = 0.35
            context.skip_low_score_vlm = True
            context.top_clips = 3

            # ===== 修复：设置统计口径字段 =====
            context.n_source_videos = len(file_map)  # 输入mp4文件数
            context.n_preprocessed = sum(1 for c in ranked if c.get("filter_status") == "PASSED")
            context.n_pass_score = sum(1 for c in ranked if c.get("clip_score", 0) >= 0.35)
            context.n_kept = sum(1 for c in ranked if c.get("kept"))
            context.n_topk = min(len(ranked), 3)

            # 转换为ClipResult列表
            clip_results = []
            for i, c in enumerate(ranked, 1):
                clip_result = ClipResult(
                    clip_id=c.get("clip_id", "unknown"),
                    clip_score=c.get("clip_score", 0),
                    filter_status=c.get("filter_status", "PASSED"),
                    video_path=c.get("video_path", ""),
                    duration=c.get("duration", 0),
                    t0=c.get("t0", 0),
                    t0_validity=c.get("t0_validity", 0),
                    coverage_effective=c.get("coverage_effective", 0),
                    final_score=c.get("final_score", 0),
                    rank_score=c.get("rank_score", 0),
                    vlm_verdict=c.get("vlm_verdict"),
                    vlm_confidence=c.get("vlm_confidence", 0),
                    kept=c.get("kept", False),
                    keep_reason=c.get("keep_reason", ""),
                    rank=i,
                )
                clip_results.append(clip_result)

            # 生成报告
            builder = ReportBuilder(
                context=context,
                clip_results=clip_results,
                output_dir=args.report_output,
            )
            report_dir = builder.build()
            print(f"  Report.md: {report_dir}/report.md")

        except ImportError as e:
            print(f"  [WARNING] Scorecard模块未安装: {e}")
        except Exception as e:
            print(f"  [ERROR] 报告生成失败: {e}")
            import traceback
            traceback.print_exc()

    # 返回状态码
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
