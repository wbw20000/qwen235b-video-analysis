"""
P0 修复实际端到端测试 — 9个车网路测FN视频
测试: POST_EVENT_ONLY逻辑 + 1280px分辨率 + OSError修复
"""
import os
import sys
import time
import json
import glob

# 确保项目根目录在 path 中
sys.path.insert(0, "d:/project2025/qwen235b")

from evaluation.evaluator import predict_file

# 9个FN视频
VIDEO_DIR = r"D:\project2025\qwen235b\uploads\大样本事故数据集\车网路测事故"
OUTPUT_DIR = r"d:\project2025\qwen235b\outputs\test_p0_real"

# 使用与消融实验相同的配置
CONFIG = {
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
    "model": "qwen3-vl-plus",
    "base_dir": os.path.join(OUTPUT_DIR, "data"),
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "data"), exist_ok=True)

    # 查找视频
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    print(f"Found {len(videos)} videos")
    print(f"image_max_width=1280, quality=80 (from config.py P0 fix)")
    print(f"model={CONFIG['model']}")
    print()

    results = []
    tp = 0
    fn = 0

    for i, vpath in enumerate(videos):
        vname = os.path.basename(vpath)
        print(f"[{i+1}/{len(videos)}] {vname}")
        t0 = time.time()

        try:
            pred = predict_file(
                video_path=vpath,
                config=CONFIG,
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
            print(f"  ERROR: {e}")
            results.append({
                "video": vname,
                "pred_label": "ERROR",
                "decision_reason": str(e),
                "elapsed": round(elapsed, 1),
                "tag": "FN",
            })

        print()

    # 汇总
    total = len(videos)
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {total}, YES(TP)={tp}, non-YES(FN)={fn}")
    print(f"Recall: {tp}/{total} = {tp/total*100:.1f}%" if total > 0 else "No videos")
    print()

    for r in results:
        print(f"  {r['tag']:>3} | {r['video'][:65]:<65} | {r['pred_label']:<12} | {r['decision_reason'][:60]}")

    # 保存结果
    out_path = os.path.join(OUTPUT_DIR, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "tp": tp,
            "fn": fn,
            "recall": tp / total if total > 0 else 0,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
