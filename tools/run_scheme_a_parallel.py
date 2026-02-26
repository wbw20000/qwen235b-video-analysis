"""
方案A 并行对比评测
- Run A：方案A管线 + Qwen3.5-35B-A3B-AWQ (G4-1, port 8000)
- Run B：方案A管线 + Qwen3-VL-32B-AWQ   (G4-2, port 8000)
同一份644视频数据，结果分别写入不同 output_dir，互不覆盖。
"""
import subprocess
import sys
import os
import time
import threading

PYTHON = r"d:/project2025/qwen235b/venv/Scripts/python.exe"
RUNNER = r"d:/project2025/qwen235b/tools/run_eval_to_output.py"

COMMON_ARGS = [
    "--acc-dir",  "D:/project2025/qwen235b/uploads/大样本事故数据集",
    "--nonacc-dir", "D:/project2025/qwen235b/uploads/大样本非交通事故数据集",
    "--acc-exclude-subdir", "非机动车违法事件视频",
    "--extra-nonacc-dir", "D:/project2025/qwen235b/uploads/大样本事故数据集/非机动车违法事件视频",
    "--dump-video-results",
    "--ablation-suppress-motion-peak",
    "--ablation-skip-metadata",
]

RUNS = [
    {
        "name": "Run-A (Qwen3.5-AWQ @ G4-1)",
        "output_dir": "outputs/scheme_a_qwen35_awq",
        "extra_args": [
            "--model", "qwen3.5-35b-a3b",
            "--vllm-url", "http://100.105.223.57:8000/v1",
        ],
        "log": "outputs/scheme_a_qwen35_awq_run.log",
    },
    {
        "name": "Run-B (Qwen3-32B-AWQ @ G4-2)",
        "output_dir": "outputs/scheme_a_qwen3_awq",
        "extra_args": [
            "--model", "qwen3-vl-32b",
            "--vllm-url", "http://100.123.59.56:8000/v1",
        ],
        "log": "outputs/scheme_a_qwen3_awq_run.log",
    },
]


def run_eval(run_cfg: dict) -> int:
    cmd = [PYTHON, RUNNER,
           "--output-dir", run_cfg["output_dir"],
           *run_cfg["extra_args"],
           *COMMON_ARGS]

    os.makedirs(run_cfg["output_dir"], exist_ok=True)
    log_path = run_cfg["log"]

    print(f"[{run_cfg['name']}] 启动...")
    print(f"  输出目录: {run_cfg['output_dir']}")
    print(f"  日志: {log_path}")

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# {run_cfg['name']}\n# 命令: {' '.join(cmd)}\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=r"d:/project2025/qwen235b")
        rc = proc.wait()

    status = "完成" if rc == 0 else f"失败(rc={rc})"
    print(f"[{run_cfg['name']}] {status}")
    return rc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["A", "B", "both"], default="both",
                        help="运行哪个 run（默认 both=并行）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印命令不执行")
    args = parser.parse_args()

    if args.dry_run:
        for r in RUNS:
            cmd = [PYTHON, RUNNER,
                   "--output-dir", r["output_dir"],
                   *r["extra_args"], *COMMON_ARGS]
            print(f"\n# {r['name']}")
            print(" ".join(cmd))
        return

    target_runs = []
    if args.run == "A":
        target_runs = [RUNS[0]]
    elif args.run == "B":
        target_runs = [RUNS[1]]
    else:
        target_runs = RUNS

    if len(target_runs) == 1:
        rc = run_eval(target_runs[0])
        sys.exit(rc)

    # 并行启动两个 run
    print("=== 方案A 并行对比评测 ===")
    print(f"Run A → {RUNS[0]['output_dir']}")
    print(f"Run B → {RUNS[1]['output_dir']}")
    print("两路同时开始...\n")

    results = {}
    threads = []

    def worker(cfg):
        results[cfg["name"]] = run_eval(cfg)

    for r in target_runs:
        t = threading.Thread(target=worker, args=(r,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(2)  # 错开启动，避免同时初始化GPU

    for t in threads:
        t.join()

    print("\n=== 评测完成 ===")
    all_ok = True
    for name, rc in results.items():
        status = "✓ 成功" if rc == 0 else f"✗ 失败(rc={rc})"
        print(f"  {name}: {status}")
        if rc != 0:
            all_ok = False

    if all_ok:
        print("\n结果目录:")
        for r in target_runs:
            summary = os.path.join(r["output_dir"], "eval", "summary.json")
            exists = "✓" if os.path.exists(summary) else "✗ (未生成)"
            print(f"  {r['output_dir']}/eval/summary.json {exists}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
