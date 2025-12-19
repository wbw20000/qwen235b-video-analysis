from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import EmbeddingConfig

try:
    import torch
    from transformers import AutoModel, AutoProcessor
except ImportError as e:  # pragma: no cover - 环境未安装时提示
    raise ImportError("缺少 transformers/torch 依赖，请先安装 requirements.txt") from e

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None


class EmbeddingIndexer:
    """
    SigLIP 编码与向量索引。
    - 支持 faiss 内存索引，缺省回退到 numpy 余弦。
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._select_device(config.device)

        # 尝试离线加载，如果失败则联网下载
        try:
            print(f"[EmbeddingIndexer] 尝试从本地缓存加载模型: {config.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                local_files_only=True  # 优先使用本地缓存
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                local_files_only=True  # 优先使用本地缓存
            )
            print(f"[EmbeddingIndexer] OK - Loaded from local cache")
        except Exception as e:
            print(f"[EmbeddingIndexer] 本地缓存不存在，开始联网下载模型...")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            print(f"[EmbeddingIndexer] OK - Model downloaded and cached")

        self.model.to(self.device)
        self.model.eval()

        # 验证模型实际所在设备
        try:
            first_param = next(self.model.parameters())
            actual_device = first_param.device
            print(f"[EmbeddingIndexer] Model actual device: {actual_device}")
            if "cuda" in str(actual_device):
                print(f"[EmbeddingIndexer] >>> GPU acceleration ENABLED!")
            else:
                print(f"[EmbeddingIndexer] WARNING: Model running on CPU, GPU not used")
        except Exception as e:
            print(f"[EmbeddingIndexer] WARNING: Cannot verify model device: {e}")

        self.vector_dim: Optional[int] = None
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.index = None

    def _select_device(self, device_pref: str) -> str:
        print(f"[EmbeddingIndexer] Device preference: {device_pref}")
        print(f"[EmbeddingIndexer] PyTorch version: {torch.__version__}")
        print(f"[EmbeddingIndexer] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[EmbeddingIndexer] CUDA device count: {torch.cuda.device_count()}")
            print(f"[EmbeddingIndexer] CUDA device name: {torch.cuda.get_device_name(0)}")

        if device_pref == "auto":
            try:
                if torch.cuda.is_available():
                    print("[EmbeddingIndexer] >>> Selected device: cuda (GPU)")
                    return "cuda"
                if torch.backends.mps.is_available():  # type: ignore
                    print("[EmbeddingIndexer] >>> Selected device: mps")
                    return "mps"
            except Exception as e:
                print(f"[EmbeddingIndexer] WARNING: Device detection error: {e}")
            print("[EmbeddingIndexer] WARNING: Fallback to CPU")
            return "cpu"
        print(f"[EmbeddingIndexer] >>> Using specified device: {device_pref}")
        return device_pref

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        total_start = time.time()
        num_images = len(image_paths)
        print(f"[SigLIP] === Image Encoding START === ({num_images} images, batch_size={self.config.batch_size})")

        # Step 1: 磁盘 I/O + PIL 预处理
        io_start = time.time()
        batches = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            batches.append(img)
        io_time = time.time() - io_start
        print(f"[SigLIP] Step 1/3: Disk I/O + PIL preprocess: {io_time:.2f}s ({num_images} images, {io_time/max(1,num_images)*1000:.1f}ms/img)")

        outputs = []
        batch_size = max(1, self.config.batch_size)
        num_batches = (len(batches) + batch_size - 1) // batch_size
        # 加固：稳健的 CUDA 检测（支持 "cuda" 和 "cuda:0" 等形式）
        use_amp = "cuda" in str(self.device)

        # Step 2: SigLIP GPU 编码
        gpu_start = time.time()
        print(f"[SigLIP] Step 2/3: GPU encoding start ({num_batches} batches, device={self.device})...")

        for batch_idx, start in enumerate(range(0, len(batches), batch_size)):
            batch_start = time.time()
            batch_imgs = batches[start : start + batch_size]

            # Processor 预处理（CPU）
            preproc_start = time.time()
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            preproc_time = time.time() - preproc_start

            # GPU 推理
            infer_start = time.time()
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    image_features = self.model.get_image_features(**inputs)
            infer_time = time.time() - infer_start

            # 加固：归一化前转 FP32，保证排序稳定性
            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # 验证：检查 NaN/Inf
            if not torch.isfinite(image_features).all():
                print("[EmbeddingIndexer] WARNING: encode_images output contains NaN/Inf!")
            outputs.append(image_features.cpu().numpy())

            batch_time = time.time() - batch_start
            print(f"[SigLIP]   Batch {batch_idx+1}/{num_batches}: {len(batch_imgs)} imgs, preproc={preproc_time:.2f}s, GPU={infer_time:.2f}s, total={batch_time:.2f}s")

        gpu_time = time.time() - gpu_start
        print(f"[SigLIP] Step 2/3: GPU encoding done: {gpu_time:.2f}s")

        # Step 3: 结果汇总
        total_time = time.time() - total_start
        print(f"[SigLIP] === Image Encoding END === Total: {total_time:.2f}s (I/O: {io_time:.2f}s, GPU: {gpu_time:.2f}s)")

        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        total_start = time.time()
        num_texts = len(texts)
        print(f"[SigLIP] === Text Encoding START === ({num_texts} templates)")

        outputs = []
        batch_size = max(1, self.config.batch_size)
        num_batches = (num_texts + batch_size - 1) // batch_size
        max_len = 64
        try:
            max_len = int(getattr(self.model.config.text_config, "max_position_embeddings", max_len))
        except Exception:
            pass
        # 加固：稳健的 CUDA 检测（支持 "cuda" 和 "cuda:0" 等形式）
        use_amp = "cuda" in str(self.device)

        for batch_idx, start in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[start : start + batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)
            # 优化：inference_mode + autocast（官方推荐写法）
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    text_features = self.model.get_text_features(**inputs)
            # 加固：归一化前转 FP32，保证排序稳定性
            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # 验证：检查 NaN/Inf
            if not torch.isfinite(text_features).all():
                print("[EmbeddingIndexer] WARNING: encode_texts output contains NaN/Inf!")
            outputs.append(text_features.cpu().numpy())

        total_time = time.time() - total_start
        print(f"[SigLIP] === Text Encoding END === Total: {total_time:.2f}s ({num_texts} templates, {num_batches} batches)")

        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def add_frame_embeddings(self, records: List[Dict]) -> List[int]:
        """
        records: [{"image_path": str, "metadata": {...}}]
        返回添加后的内部 ID 列表。
        """
        if not records:
            return []

        image_paths = [r["image_path"] for r in records]
        embeddings = self.encode_images(image_paths).astype(np.float32)
        if embeddings.size == 0:
            return []

        if self.vector_dim is None:
            self.vector_dim = embeddings.shape[1]
            if faiss is not None:
                self.index = faiss.IndexFlatIP(self.vector_dim)

        ids = []
        for emb, record in zip(embeddings, records):
            idx = len(self.embeddings)
            self.embeddings.append(emb)
            self.metadata.append(record["metadata"])
            ids.append(idx)

        if self.index is not None:
            self.index.add(embeddings)
        return ids

    def _search_with_faiss(self, text_embeds: np.ndarray, top_m: int) -> List[List[Tuple[int, float]]]:
        if self.index is None or text_embeds.size == 0:
            return [[] for _ in range(len(text_embeds))]
        scores, idxs = self.index.search(text_embeds, top_m)
        results: List[List[Tuple[int, float]]] = []
        for score_row, idx_row in zip(scores, idxs):
            pairs = []
            for i, s in zip(idx_row, score_row):
                if i >= 0:
                    pairs.append((int(i), float(s)))
            results.append(pairs)
        return results

    def _search_with_numpy(self, text_embeds: np.ndarray, top_m: int) -> List[List[Tuple[int, float]]]:
        if not self.embeddings:
            return [[] for _ in range(len(text_embeds))]
        frame_matrix = np.vstack(self.embeddings)  # (N, D)
        results = []
        for t in text_embeds:
            sims = frame_matrix @ t
            top_idx = np.argsort(-sims)[:top_m]
            results.append([(int(i), float(sims[i])) for i in top_idx])
        return results

    def search_by_text_embeddings(self, text_embeds: np.ndarray, top_m: int) -> List[List[Tuple[int, float]]]:
        if self.index is not None:
            return self._search_with_faiss(text_embeds, top_m)
        return self._search_with_numpy(text_embeds, top_m)

    def multi_template_search(self, templates: List[str]) -> List[Dict]:
        """
        多模板检索：每个模板取 top_M，然后对同一帧保留最大相似度。
        返回 frame_candidates_topN。
        """
        if not templates:
            return []

        text_embeds = self.encode_texts(templates)
        top_m = self.config.top_m_per_template
        per_template_results = self.search_by_text_embeddings(text_embeds, top_m)

        best_map: Dict[int, Dict] = {}
        for tmpl_idx, results in enumerate(per_template_results):
            for frame_id, score in results:
                if frame_id not in best_map or score > best_map[frame_id]["similarity_score"]:
                    best_map[frame_id] = {
                        "frame_id": frame_id,
                        "similarity_score": score,
                        "template_index": tmpl_idx,
                        "metadata": self.metadata[frame_id],
                    }

        # 截断前 frame_top_n
        top_items = sorted(best_map.values(), key=lambda x: x["similarity_score"], reverse=True)
        return top_items[: self.config.frame_top_n]
