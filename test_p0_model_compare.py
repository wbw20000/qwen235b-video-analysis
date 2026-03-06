"""
P0 修复三模型对比测试 — 16个车网路测事故视频
模型: qwen3-vl-32b-instruct (DashScope) / qwen3.5-35b-a3b (g4-1 vLLM)
对比: 已有 qwen3-vl-plus 结果
"""
import os
import sys
import time
import json
import glob
import argparse

sys.path.insert(0, "d:/project2025/qwen235b")

from evaluation.evaluator import predict_file

VIDEO_DIR = r"D:\project2025\qwen235b\uploads\大样本事故数据集\车网路测事故"

# 两种模型配置
MODEL_CONFIGS = {
    "32b-instruct": {
        "model": "qwen3-vl-32b-instruct",
        "output_dir": r"d:\project2025\qwen235b\outputs\test_p0_32b_instruct",
        "env": {},  # 使用 DashScope
    },
    "qwen35": {
        "model": "qwen3.5-35b-a3b",
        "output_dir": r"d:\project2025\qwen235b\outputs\test_p0_qwen35",
        "env": {
            "VLLM_BASE_URL": "http://100.105.223.57:8000/v1",
            "VLLM_MODEL_NAME": "qwen3.5-35b-a3b",
        },
    },
}

BASE_CONFIG = {
    "clip_score_threshold": 0.35,
    "top_clips": 7,
    "B_full_process": 0.2,
    "P_post_event": 0.2,
    "min_pre_sec": 3,
    "min_post_sec": 5,
    "pre_roll": 8.0,
    "post_roll": 12.0,
    "enable_top1_fallback": False,
    "enable_narrative_sidecar": False,
    "narrative_trigger_yes": True,
    "narrative_trigger_uncertain": False,
    "narrative_trigger_post_event": False,
}


def run_test(model_key):
    mc = MODEL_CONFIGS[model_key]
    output_dir = mc["output_dir"]
    data_dir = os.path.join(output_dir, "data")

    # 设置环境变量
    old_env = {}
    for k, v in mc["env"].items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    # 清除可能冲突的环境变量
    if not mc["env"].get("VLLM_BASE_URL"):
        # DashScope 模式，确保没有 VLLM_BASE_URL 干扰
        if "VLLM_BASE_URL" in os.environ:
            old_env["VLLM_BASE_URL"] = os.environ.pop("VLLM_BASE_URL")

    os.makedirs(data_dir, exist_ok=True)

    config = {**BASE_CONFIG, "model": mc["model"], "base_dir": data_dir}

    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"\n{'='*80}")
    print(f"MODEL: {mc['model']} ({model_key})")
    print(f"Videos: {len(videos)}, Output: {output_dir}")
    print(f"{'='*80}\n")

    results = []
    tp = fn = 0

    for i, vpath in enumerate(videos):
        vname = os.path.basename(vpath)
        print(f"[{i+1}/{len(videos)}] {vname}")
        t0 = time.time()

        try:
            pred = predict_file(
                video_path=vpath,
                config=config,
                user_query="检测交通事故",
            )
            elapsed = time.time() - t0
            label = pred.pred_label
            reason = pred.decision_reason

            if label == "YES":
                tp += 1
                tag = "TP"
            else:
                fn += 1
                tag = "FN"

            print(f"  {tag} | {label} | {reason} | {elapsed:.1f}s")
            results.append({
                "video": vname,
                "pred_label": label,
                "decision_reason": reason,
                "elapsed": round(elapsed, 1),
                "tag": tag,
            })

        except Exception as e:
            elapsed = time.time() - t0
            fn += 1
            print(f"  ERROR: {e} | {elapsed:.1f}s")
            results.append({
                "video": vname,
                "pred_label": "ERROR",
                "decision_reason": str(e),
                "elapsed": round(elapsed, 1),
                "tag": "FN",
            })

        print()

    # 恢复环境变量
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    # 汇总
    total = len(videos)
    print("=" * 80)
    print(f"SUMMARY — {mc['model']}")
    print("=" * 80)
    print(f"Total: {total}, TP={tp}, FN={fn}")
    recall = tp / total * 100 if total > 0 else 0
    print(f"Recall: {tp}/{total} = {recall:.1f}%")
    print()

    for r in results:
        print(f"  {r['tag']:>3} | {r['video'][:60]:<60} | {r['pred_label']:<12} | {r['elapsed']}s")

    # 保存
    out_path = os.path.join(output_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": mc["model"],
            "total": total,
            "tp": tp,
            "fn": fn,
            "recall": tp / total if total > 0 else 0,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {out_path}")
    return tp, fn, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    run_test(args.model)
