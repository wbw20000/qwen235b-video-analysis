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
    parser.add_argument('--ablation-skip-siglip', action='store_true',
                        help='消融测试：跳过SigLIP编码和检索')
    parser.add_argument('--model', type=str, default=None,
                        help='VLM模型名 (如 qwen3-vl-8b-instruct)')
    parser.add_argument('--reuse-preprocess', type=str, default=None,
                        help='复用预处理缓存的源base_dir (如 data)')
    # 消融 v2 参数
    parser.add_argument('--ablation-skip-metadata', action='store_true',
                        help='消融测试：不传 metadata_text 给 VLM')
    parser.add_argument('--ablation-disable-yolo', action='store_true',
                        help='消融测试：禁用 YOLO 检测+跟踪')
    parser.add_argument('--ablation-force-uniform', action='store_true',
                        help='消融测试：强制均匀帧选择（替代信号驱动选帧）')
    parser.add_argument('--ablation-skip-mog2', action='store_true',
                        help='消融测试：跳过 MOG2 运动检测')
    parser.add_argument('--ablation-suppress-motion-peak', action='store_true',
                        help='方案A：压制 motion_peak 选帧信号（权重清零），保留其他信号')
    parser.add_argument('--enable-accident-rag', action='store_true',
                        help='Phase B: 开启 Image RAG 视觉样本注入（需先确认 data/accident_exemplars/ 帧内容）')
    parser.add_argument('--accident-rag-dir', type=str, default=None,
                        help='Image RAG exemplar 目录（默认 data/accident_exemplars）')
    parser.add_argument('--accident-rag-top-k', type=int, default=2,
                        help='每次注入的 exemplar 帧数（默认 2）')
    parser.add_argument('--vllm-url', type=str, default=None,
                        help='本地 vLLM 端点 URL（如 http://100.105.223.57:8000/v1），覆盖 VLLM_BASE_URL 环境变量')
    # 数据集处理参数
    parser.add_argument('--acc-exclude-subdir', type=str, action='append', default=[],
                        help='从事故目录中排除的子目录名（可多次指定）')
    parser.add_argument('--extra-nonacc-dir', type=str, action='append', default=[],
                        help='额外的非事故数据目录（可多次指定）')
    parser.add_argument('--shard-id', type=int, default=None,
                        help='分片ID（从0开始），需配合 --num-shards 使用')
    parser.add_argument('--num-shards', type=int, default=None,
                        help='总分片数，用于多进程并行评测')
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

    # 消融测试配置
    if args.ablation_skip_siglip:
        config["ablation_skip_siglip"] = True
        print(f"  [消融] 跳过SigLIP: 启用")
    if args.ablation_skip_metadata:
        config["ablation_skip_metadata"] = True
        print(f"  [消融] 跳过Metadata文本: 启用")
    if args.ablation_disable_yolo:
        config["ablation_disable_yolo"] = True
        print(f"  [消融] 禁用YOLO检测: 启用")
    if args.ablation_force_uniform:
        config["ablation_force_uniform"] = True
        print(f"  [消融] 强制均匀帧选择: 启用")
    if args.ablation_skip_mog2:
        config["ablation_skip_mog2"] = True
        print(f"  [消融] 跳过MOG2运动检测: 启用")
    if args.ablation_suppress_motion_peak:
        config["ablation_suppress_motion_peak"] = True
        print(f"  [方案A] 压制motion_peak信号: 启用")

    # VLM模型切换
    if args.model:
        config["model"] = args.model
        print(f"  VLM模型: {args.model}")

    # 预处理缓存复用
    if args.reuse_preprocess:
        config["reuse_preprocess_dir"] = args.reuse_preprocess
        print(f"  复用预处理: {args.reuse_preprocess}")

    # Phase B: Image RAG
    if args.enable_accident_rag:
        config["accident_rag_enabled"] = True
        config["accident_rag_top_k"] = args.accident_rag_top_k
        if args.accident_rag_dir:
            config["accident_rag_exemplars_dir"] = args.accident_rag_dir
        print(f"  [Phase B] Image RAG: 启用 (top_k={args.accident_rag_top_k})")

    # vLLM 端点覆盖（优先级高于 VLLM_BASE_URL 环境变量）
    if args.vllm_url:
        os.environ["VLLM_BASE_URL"] = args.vllm_url
        print(f"  vLLM端点: {args.vllm_url}")

    # 中间数据隔离：存到 output_dir/data/ 下，避免不同模型互相覆盖
    data_base_dir = os.path.join(output_dir, "data")
    config["base_dir"] = data_base_dir
    print(f"  中间数据目录: {data_base_dir}")

    # 数据集路径
    acc_dir = args.acc_dir or "D:/project2025/qwen235b/uploads/事故数据集"
    nonacc_dir = args.nonacc_dir or "D:/project2025/qwen235b/uploads/非交通事故数据集"

    if args.acc_exclude_subdir:
        print(f"  事故目录排除子目录: {args.acc_exclude_subdir}")
    if args.extra_nonacc_dir:
        print(f"  额外非事故目录: {args.extra_nonacc_dir}")

    # 创建评测器
    evaluator = Evaluator(
        acc_dir=acc_dir,
        nonacc_dir=nonacc_dir,
        config=config,
        output_dir=output_dir,
        subset_per_class=args.subset,  # 0=全量, >0=随机采样
        seed=42,
        dump_video_results=args.dump_video_results,
        acc_exclude_subdirs=args.acc_exclude_subdir,
        extra_nonacc_dirs=args.extra_nonacc_dir,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
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
