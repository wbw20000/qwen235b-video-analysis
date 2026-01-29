#!/bin/bash
# 全量评测脚本 - 使用本地 vLLM (离线模式)

set -e

# 强制离线模式 - 不访问 HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置环境变量指向本地 vLLM
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="EMPTY"

# SigLIP 模型路径 (使用绝对本地路径)
export SIGLIP_MODEL_PATH="/data/models/siglip-base-patch16-384"

# 激活虚拟环境
source /data/app/venv/bin/activate

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
OUTPUT_DIR="outputs/k8s_full_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p /data/app/$OUTPUT_DIR

echo "========================================"
echo "  全量评测开始 (离线模式)"
echo "  输出目录: $OUTPUT_DIR"
echo "  vLLM: $VLLM_BASE_URL"
echo "  SigLIP: $SIGLIP_MODEL_PATH"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "========================================"

# 运行评测
cd /data/app
python tools/run_eval_to_output.py   --output-dir $OUTPUT_DIR   --acc-dir /data1/testdata/vaildata/大样本事故数据集   --nonacc-dir /data1/testdata/vaildata/大样本非交通事故数据集   --dump-video-results

echo "评测完成! 结果保存在: $OUTPUT_DIR"
