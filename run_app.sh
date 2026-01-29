#!/bin/bash
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL_NAME=qwen3-vl-32b
export CUDA_VISIBLE_DEVICES=0,1

cd /data/app
source venv/bin/activate
python app.py --host 0.0.0.0 --port 5000
