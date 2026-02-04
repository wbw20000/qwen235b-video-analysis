#!/usr/bin/env python3
"""
Embedding Service v17 - 与 Windows 版本一致的实现
- 使用 transformers 4.57.3 (与 Windows 一致)
- 使用 AutoModel/AutoProcessor
- get_image_features/get_text_features 直接返回张量
"""
import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

from workers.common.logging_config import setup_logger, LogContext

# 配置
MODEL_NAME = os.getenv("MODEL_NAME", "google/siglip-base-patch16-384")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/siglip-base-patch16-384")
MAX_REQUESTS_BEFORE_EXIT = int(os.getenv("MAX_REQUESTS", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

app = Flask(__name__)
logger = setup_logger("embedding-service")
log = LogContext(logger, stage="embedding")


class EmbeddingModel:
    """SigLIP 模型封装 - 与 Windows embedding_indexer.py 一致"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = None
        self.request_count = 0

    def load(self):
        """加载模型"""
        from transformers import AutoModel, AutoProcessor
        import transformers

        log.info(f"加载模型: {self.model_path}")
        log.info(f"transformers 版本: {transformers.__version__}")

        # 选择设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            log.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            log.warning("GPU 不可用，使用 CPU")

        # 使用 AutoModel 和 AutoProcessor (与 Windows 版本一致)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        log.info("模型加载完成 (v17 - transformers 4.57.3)")

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        批量编码图像为向量
        使用 model.get_image_features() (与 Windows 一致)
        返回: (N, D) 的 numpy 数组，已 L2 归一化
        """
        if not images:
            return np.array([])

        self.request_count += 1
        outputs = []
        use_amp = "cuda" in str(self.device)

        with torch.inference_mode():
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]

                # 使用 processor 处理图像
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 使用 AMP 加速 (与 Windows 一致)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    image_features = self.model.get_image_features(**inputs)

                # L2 归一化 (与 Windows 一致)
                image_features = image_features.float()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                outputs.append(image_features.cpu().numpy())

                # 清理
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return np.concatenate(outputs, axis=0) if outputs else np.array([])

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本为向量
        使用 model.get_text_features() (与 Windows 一致)
        返回: (N, D) 的 numpy 数组，已 L2 归一化
        """
        if not texts:
            return np.array([])

        outputs = []
        use_amp = "cuda" in str(self.device)

        # 获取 max_length (与 Windows 一致)
        max_len = 64
        try:
            max_len = int(getattr(self.model.config.text_config, "max_position_embeddings", max_len))
        except Exception:
            pass

        with torch.inference_mode():
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]

                # 使用 processor 处理文本
                inputs = self.processor(
                    text=batch_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 使用 AMP 加速 (与 Windows 一致)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    text_features = self.model.get_text_features(**inputs)

                # L2 归一化 (与 Windows 一致)
                text_features = text_features.float()
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                outputs.append(text_features.cpu().numpy())

                # 清理
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return np.concatenate(outputs, axis=0) if outputs else np.array([])


# 全局模型实例
model: Optional[EmbeddingModel] = None


def init_model():
    """初始化模型"""
    global model
    path = MODEL_PATH if Path(MODEL_PATH).exists() else MODEL_NAME
    model = EmbeddingModel(path)
    model.load()


@app.route("/health", methods=["GET"])
def health():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "version": "v17",
        "model": MODEL_NAME,
        "device": str(model.device) if model else "not loaded",
        "request_count": model.request_count if model else 0
    })


@app.route("/encode/images", methods=["POST"])
def encode_images():
    """编码图像列表"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data or "images" not in data:
        return jsonify({"error": "Missing 'images' field"}), 400

    job_id = data.get("job_id", "")
    trace_id = data.get("trace_id", "")

    try:
        images = []
        for b64_img in data["images"]:
            img_bytes = base64.b64decode(b64_img)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(img)

        log.info(f"编码 {len(images)} 张图像", job_id=job_id, trace_id=trace_id)
        embeddings = model.encode_images(images)

        if model.request_count >= MAX_REQUESTS_BEFORE_EXIT:
            log.info(f"达到请求阈值 ({MAX_REQUESTS_BEFORE_EXIT})，标记退出")

        return jsonify({
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape)
        })

    except Exception as e:
        log.error(f"编码失败: {e}", job_id=job_id, trace_id=trace_id)
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/encode/text", methods=["POST"])
def encode_text():
    """编码文本列表"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    try:
        texts = data["texts"]
        log.info(f"编码 {len(texts)} 条文本")
        embeddings = model.encode_text(texts)

        return jsonify({
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape)
        })

    except Exception as e:
        log.error(f"编码失败: {e}")
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/similarity", methods=["POST"])
def compute_similarity():
    """计算图像与文本的相似度"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json
    if not data or "images" not in data or "texts" not in data:
        return jsonify({"error": "Missing 'images' or 'texts' field"}), 400

    job_id = data.get("job_id", "")
    trace_id = data.get("trace_id", "")

    try:
        images = []
        for b64_img in data["images"]:
            img_bytes = base64.b64decode(b64_img)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            images.append(img)

        texts = data["texts"]

        log.info(f"计算相似度: {len(images)} 图像 x {len(texts)} 文本", job_id=job_id, trace_id=trace_id)

        # 编码 (已经 L2 归一化)
        img_embeddings = model.encode_images(images)
        text_embeddings = model.encode_text(texts)

        # 计算余弦相似度 (由于已归一化，直接点积)
        similarities = np.dot(img_embeddings, text_embeddings.T)

        return jsonify({
            "similarities": similarities.tolist(),
            "shape": list(similarities.shape)
        })

    except Exception as e:
        log.error(f"相似度计算失败: {e}", job_id=job_id, trace_id=trace_id)
        import traceback
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embedding Service v17")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    args = parser.parse_args()

    init_model()
    log.info(f"启动 Embedding Service v17: {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
