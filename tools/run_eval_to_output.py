# -*- coding: utf-8 -*-
"""
评测脚本 - 输出到指定目录

用法:
    python tools/run_eval_to_output.py --output-dir outputs/run_current
    python tools/run_eval_to_output.py --output-dir outputs/run_fallback
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 设置编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description='42视频完整评测')
    parser.add_argument('--output-dir', '-o', required=True, help='输出目录')
    parser.add_argument('--dump-video-results', action='store_true',
                        help='保存每个视频的完整result到video_results/')
    parser.add_argument('--enable-fallback', action='store_true',
                        help='启用Top-1 Fallback (无clip通过阈值时强制送入top1)')
    # Sidecar 参数
    parser.add_argument('--enable-narrative-sidecar', action='store_true',
                        help='启用事故复盘描述Sidecar')
    parser.add_argument('--narrative-trigger-yes', action='store_true', default=True,
                        help='verdict=YES时触发Sidecar (默认启用)')
    parser.add_argument('--narrative-trigger-uncertain', action='store_true',
                        help='verdict=UNCERTAIN时触发Sidecar')
    parser.add_argument('--narrative-trigger-post-event', action='store_true',
                        help='verdict=POST_EVENT_ONLY时触发Sidecar')
    # 自定义数据集路径
    parser.add_argument('--acc-dir', type=str, default=None,
                        help='事故数据集目录 (默认: uploads/事故数据集)')
    parser.add_argument('--nonacc-dir', type=str, default=None,
                        help='非事故数据集目录 (默认: uploads/非交通事故数据集)')
    parser.add_argument('--subset', type=int, default=0,
                        help='每类随机采样数量 (0=全量)')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    fallback_mode = "启用" if args.enable_fallback else "禁用"
    sidecar_mode = "启用" if args.enable_narrative_sidecar else "禁用"

    print("=" * 70)
    print("  Full Eval - 全量测试")
    print("=" * 70)
    print(f"  时间: {datetime.now().isoformat()}")
    print(f"  输出目录: {output_dir}")
    print(f"  Top-1 Fallback: {fallback_mode}")
    print(f"  Narrative Sidecar: {sidecar_mode}")
    if args.enable_narrative_sidecar:
        triggers = []
        if args.narrative_trigger_yes:
            triggers.append("YES")
        if args.narrative_trigger_uncertain:
            triggers.append("UNCERTAIN")
        if args.narrative_trigger_post_event:
            triggers.append("POST_EVENT_ONLY")
        print(f"    触发条件: {', '.join(triggers) if triggers else 'YES'}")
    print()

    # 配置
    config = {
        "clip_score_threshold": 0.35,
        "top_clips": 7,
        "B_full_process": 0.2,
        "P_post_event": 0.2,
        "min_pre_sec": 3,
        "min_post_sec": 5,
        "pre_roll": 8.0,
        "post_roll": 12.0,
        "enable_top1_fallback": args.enable_fallback,
        # Sidecar 配置
        "enable_narrative_sidecar": args.enable_narrative_sidecar,
        "narrative_trigger_yes": args.narrative_trigger_yes,
        "narrative_trigger_uncertain": args.narrative_trigger_uncertain,
        "narrative_trigger_post_event": args.narrative_trigger_post_event,
    }

    # 数据集路径
    acc_dir = args.acc_dir or "D:/project2025/qwen235b/uploads/事故数据集"
    nonacc_dir = args.nonacc_dir or "D:/project2025/qwen235b/uploads/非交通事故数据集"

    # 创建评测器
    evaluator = Evaluator(
        acc_dir=acc_dir,
        nonacc_dir=nonacc_dir,
        config=config,
        output_dir=output_dir,
        subset_per_class=args.subset,  # 0=全量, >0=随机采样
        seed=42,
        dump_video_results=args.dump_video_results,
    )

    # 运行评测
    eval_id = "eval"
    result = evaluator.run(eval_id=eval_id)

    # 获取指标
    metrics_dual = result.get("metrics_dual", {})
    strict = metrics_dual.get("strict", result.get("metrics", {}))
    conservative = metrics_dual.get("conservative", {})

    # 生成metrics.md
    metrics_md_path = os.path.join(output_dir, "eval", "metrics.md")
    os.makedirs(os.path.dirname(metrics_md_path), exist_ok=True)

    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write("# 评测指标报告\n\n")
        f.write(f"生成时间: {datetime.now().isoformat()}\n\n")

        f.write("## STRICT模式\n\n")
        f.write(f"| 指标 | 值 |\n")
        f.write(f"|------|----|\n")
        f.write(f"| TP | {strict.get('tp', 0)} |\n")
        f.write(f"| FP | {strict.get('fp', 0)} |\n")
        f.write(f"| TN | {strict.get('tn', 0)} |\n")
        f.write(f"| FN | {strict.get('fn', 0)} |\n")
        f.write(f"| Recall | {strict.get('recall', 0):.4f} |\n")
        f.write(f"| FPR | {strict.get('fpr', 0):.4f} |\n")
        f.write(f"| Precision | {strict.get('precision', 0):.4f} |\n")
        f.write(f"| F1 | {strict.get('f1', 0):.4f} |\n")
        f.write(f"| Accuracy | {strict.get('accuracy', 0):.4f} |\n")
        f.write(f"| Abstain | {strict.get('abstain', 0)} |\n")
        f.write(f"\n")

        label_dist = strict.get('label_dist', {})
        f.write("### 标签分布\n\n")
        f.write(f"- YES: {label_dist.get('YES', 0)}\n")
        f.write(f"- NO: {label_dist.get('NO', 0)}\n")
        f.write(f"- UNCERTAIN: {label_dist.get('UNCERTAIN', 0)}\n")
        f.write(f"\n")

        if conservative:
            f.write("## CONSERVATIVE模式\n\n")
            f.write(f"| 指标 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| TP | {conservative.get('tp', 0)} |\n")
            f.write(f"| FP | {conservative.get('fp', 0)} |\n")
            f.write(f"| TN | {conservative.get('tn', 0)} |\n")
            f.write(f"| FN | {conservative.get('fn', 0)} |\n")
            f.write(f"| Recall | {conservative.get('recall', 0):.4f} |\n")
            f.write(f"| FPR | {conservative.get('fpr', 0):.4f} |\n")
            f.write(f"| Precision | {conservative.get('precision', 0):.4f} |\n")
            f.write(f"| F1 | {conservative.get('f1', 0):.4f} |\n")

        # 检查ENGINE_ERROR
        f.write("\n## 错误统计\n\n")
        per_file_path = os.path.join(output_dir, "eval", "per_file.json")
        if os.path.exists(per_file_path):
            with open(per_file_path, 'r', encoding='utf-8') as pf:
                per_file = json.load(pf)

            error_types = {}
            for item in per_file:
                if item.get('error'):
                    error_str = item['error']
                    # 提取错误类型
                    if 'Errno 22' in error_str:
                        error_type = 'Errno 22 (Invalid argument)'
                    elif 'MemoryError' in error_str:
                        error_type = 'MemoryError'
                    elif 'TimeoutError' in error_str:
                        error_type = 'TimeoutError'
                    else:
                        # 提取第一行异常类型
                        lines = error_str.strip().split('\n')
                        error_type = lines[-1][:50] if lines else 'Unknown'

                    error_types[error_type] = error_types.get(error_type, 0) + 1

            if error_types:
                f.write("| 错误类型 | 数量 |\n")
                f.write("|----------|------|\n")
                for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
                    f.write(f"| {etype} | {count} |\n")
            else:
                f.write("*无错误*\n")

    print(f"\n评测完成！")
    print(f"输出目录: {os.path.join(output_dir, 'eval')}")
    print(f"指标报告: {metrics_md_path}")

    # 打印关键指标
    print("\n" + "=" * 50)
    print("STRICT模式关键指标:")
    print(f"  TP={strict.get('tp', 0)}, FP={strict.get('fp', 0)}, TN={strict.get('tn', 0)}, FN={strict.get('fn', 0)}")
    print(f"  Recall={strict.get('recall', 0):.4f}, FPR={strict.get('fpr', 0):.4f}")
    print(f"  Precision={strict.get('precision', 0):.4f}, F1={strict.get('f1', 0):.4f}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
