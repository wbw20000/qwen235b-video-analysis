#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0

cd /data/app
source venv/bin/activate

python -m vllm.entrypoints.openai.api_server   --model /data/models/tclf90/Qwen3-VL-32B-Instruct-AWQ   --served-model-name qwen3-vl-32b   --tensor-parallel-size 2   --max-model-len 4096   --gpu-memory-utilization 0.85   --max-num-seqs 2   --port 8000   --trust-remote-code   --enforce-eager
