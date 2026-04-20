#!/bin/bash
# VLM 启动脚本 - 固定使用 GPU 4,5
# 用法: ./start_vllm.sh

set -e

# 固定 GPU 4,5 (缩减为 TP=2)
export CUDA_VISIBLE_DEVICES=4,5
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

# 日志目录
LOG_DIR=/data/logs
mkdir -p $LOG_DIR

# 检查是否已经运行
if pgrep -f "vllm.entrypoints.openai.api_server" > /dev/null; then
    echo "vLLM 已经在运行，先停止..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
fi

echo "启动 vLLM (GPU 4,5, TP=2, max-seqs=4)..."
cd /data/app
source venv/bin/activate

nohup python -m vllm.entrypoints.openai.api_server   --model /data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ   --served-model-name qwen3-vl-32b   --tensor-parallel-size 2   --max-model-len 4096   --gpu-memory-utilization 0.85   --max-num-seqs 4   --port 8000   --trust-remote-code   --enforce-eager   > $LOG_DIR/vllm.log 2>&1 &

sleep 5
echo "vLLM 启动完成，PID: $(pgrep -f 'vllm.entrypoints.openai.api_server')"
echo "日志: $LOG_DIR/vllm.log"
