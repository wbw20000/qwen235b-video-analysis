#!/bin/bash
# G4-1 一键部署 Qwen3.5-35B-A3B-AWQ
# 在 G4-1 上执行：bash /data/setup_qwen35.sh
set -e

VENV=/data/venvs/vllm
MODEL_DIR=/data/models/Qwen3.5-35B-A3B-AWQ
LOG_DIR=/data/logs
mkdir -p $LOG_DIR

echo "=== [1/4] 检查 vLLM nightly 安装状态 ==="
$VENV/bin/python -m vllm --version 2>/dev/null || $VENV/bin/pip show vllm | grep Version
# 确认支持 Qwen3.5
$VENV/bin/python -c "
from vllm.model_executor.models import ModelRegistry
archs = ModelRegistry.get_supported_archs()
has35 = any('Qwen3_5' in a for a in archs)
print('Qwen3_5MoeForConditionalGeneration 支持:', has35)
if not has35:
    print('ERROR: 请先安装 vLLM nightly')
    import sys; sys.exit(1)
"

echo ""
echo "=== [2/4] 下载 Qwen3.5-35B-A3B-AWQ (float16, ~17.5GB) ==="
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "模型已存在，跳过下载"
    du -sh $MODEL_DIR
else
    $VENV/bin/python -c "
from modelscope import snapshot_download
snapshot_download('QuantTrio/Qwen3.5-35B-A3B-AWQ', local_dir='$MODEL_DIR')
print('下载完成')
"
fi

echo ""
echo "=== [3/4] 停止旧 vLLM 进程 ==="
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null && echo "已停止" || echo "无运行中进程"
sleep 3

echo ""
echo "=== [4/4] 启动 Qwen3.5-35B-A3B-AWQ (8x T4, TP=8, port=8000) ==="
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

nohup $VENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_DIR \
    --served-model-name qwen3.5-35b-a3b \
    --tensor-parallel-size 8 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 4 \
    --port 8000 \
    --dtype half \
    --enforce-eager \
    --trust-remote-code \
    > $LOG_DIR/vllm_qwen35_awq.log 2>&1 &

VPID=$!
echo "vLLM PID: $VPID"
echo "等待启动 (60s)..."
sleep 60
if kill -0 $VPID 2>/dev/null; then
    echo "进程存活，查看最后日志:"
    tail -20 $LOG_DIR/vllm_qwen35_awq.log
else
    echo "ERROR: 进程已退出，查看日志:"
    tail -30 $LOG_DIR/vllm_qwen35_awq.log
fi
