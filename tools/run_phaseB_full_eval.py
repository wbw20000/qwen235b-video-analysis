"""
run_phaseB_full_eval.py
Phase B (Prompt RAG + Image RAG + No-Metadata) 全数据集评测
串行跑三个模型，共用 ablation_v2 C0 instruct 预处理缓存

Run 1: qwen3-vl-32b     (G4-2 本地 vLLM, AWQ)
Run 2: qwen3.5-27b      (阿里云 DashScope 云端)
Run 3: qwen3.5-35b-a3b  (阿里云 DashScope 云端)
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

PYTHON = r"d:/project2025/qwen235b/venv/Scripts/python.exe"
EVAL_SCRIPT = r"d:/project2025/qwen235b/tools/run_eval_to_output.py"
BASE_DIR = Path(r"d:/project2025/qwen235b")

ACC_DIR = r"D:/project2025/qwen235b/uploads/大样本事故数据集"
NONACC_DIR = r"D:/project2025/qwen235b/uploads/大样本非交通事故数据集"
ACC_EXCLUDE = "非机动车违法事件视频"
EXTRA_NONACC = r"D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频"

RUNS = [
    {
        "name": "Run1_phaseB_qwen3vl32b_awq",
        "model": "qwen3-vl-32b",
        "output_dir": "outputs/phaseB_nomd_awq",
        "reuse_preprocess": "outputs/ablation_v2_C0_32b_instruct/data",
        "vllm_url": "http://100.123.59.56:8000/v1",
    },
    {
        "name": "Run2_phaseB_qwen3527b_cloud",
        "model": "qwen3.5-27b",
        "output_dir": "outputs/phaseB_nomd_qwen35_27b",
        "reuse_preprocess": "outputs/ablation_v2_C0_32b_instruct/data",
        "vllm_url": None,  # DashScope 云端
    },
    {
        "name": "Run3_phaseB_qwen3535b_cloud",
        "model": "qwen3.5-35b-a3b",
        "output_dir": "outputs/phaseB_nomd_qwen35_35b",
        "reuse_preprocess": "outputs/ablation_v2_C0_32b_instruct/data",
        "vllm_url": None,  # DashScope 云端
    },
]


def summary_exists(output_dir: str) -> bool:
    p = BASE_DIR / output_dir / "eval" / "summary.json"
    return p.exists()


def run_eval(run: dict) -> bool:
    name = run["name"]
    output_dir = run["output_dir"]

    if summary_exists(output_dir):
        print(f"\n[SKIP] {name} — summary.json 已存在，跳过")
        return True

    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--output-dir", output_dir,
        "--model", run["model"],
        "--acc-dir", ACC_DIR,
        "--nonacc-dir", NONACC_DIR,
        "--acc-exclude-subdir", ACC_EXCLUDE,
        "--extra-nonacc-dir", EXTRA_NONACC,
        "--reuse-preprocess", run["reuse_preprocess"],
        "--enable-accident-rag",
        "--accident-rag-top-k", "2",
        "--ablation-skip-metadata",
        "--dump-video-results",
    ]
    if run.get("vllm_url"):
        cmd += ["--vllm-url", run["vllm_url"]]

    env = os.environ.copy()
    # 云端模式确保不带 VLLM_BASE_URL
    if not run.get("vllm_url"):
        env.pop("VLLM_BASE_URL", None)

    log_path = BASE_DIR / f"{output_dir}_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[START] {name}")
    print(f"  模型: {run['model']}")
    print(f"  输出: {output_dir}")
    print(f"  复用预处理: {run['reuse_preprocess']}")
    print(f"  日志: {log_path}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(BASE_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode == 0 and summary_exists(output_dir):
        summary = json.loads((BASE_DIR / output_dir / "eval" / "summary.json").read_text(encoding="utf-8"))
        m = summary["metrics"]
        print(f"\n[DONE] {name}")
        print(f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']} abstain={m.get('abstain',0)}")
        print(f"  Recall={m['recall']:.1%}  FPR={m['fpr']:.1%}  F1={m['f1']:.1%}")
        return True
    else:
        print(f"\n[FAIL] {name}  returncode={proc.returncode}")
        print(f"  查看日志: {log_path}")
        return False


def main():
    print(f"Phase B 全数据集评测批量脚本")
    print(f"共 {len(RUNS)} 个 runs，串行执行")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = []
    for run in RUNS:
        ok = run_eval(run)
        results.append((run["name"], ok))

    print(f"\n{'='*70}")
    print("全部完成:")
    for name, ok in results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}]  {name}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
