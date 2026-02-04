#!/bin/bash
# VLM 启动脚本 - 固定使用 GPU 6,7
# 用法: ./start_vllm.sh

set -e

# 固定 GPU 6,7
export CUDA_VISIBLE_DEVICES=6,7
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

echo "启动 vLLM (GPU 6,7, TP=2)..."
cd /data/app
source venv/bin/activate

nohup python -m vllm.entrypoints.openai.api_server \
  --model /data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name qwen3-vl-32b \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2 \
  --port 8000 \
  --trust-remote-code \
  --enforce-eager \
  > $LOG_DIR/vllm.log 2>&1 &

echo "vLLM 已启动，PID: $!"
echo "日志: $LOG_DIR/vllm.log"
echo "GPU: 6,7 (TP=2)"
