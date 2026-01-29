#\!/bin/bash
# vLLM 启动脚本 - T4 GPU 兼容模式

export CUDA_VISIBLE_DEVICES=2,3
export CUDA_HOME=/usr
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /data/app
source venv/bin/activate

nohup python -m vllm.entrypoints.openai.api_server   --model /data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ   --served-model-name qwen3-vl-32b   --tensor-parallel-size 2   --max-model-len 4096   --gpu-memory-utilization 0.85   --max-num-seqs 2   --port 8000   --trust-remote-code   --enforce-eager   > /data/logs/vllm.log 2>&1 &

echo "vLLM PID: $\!"
