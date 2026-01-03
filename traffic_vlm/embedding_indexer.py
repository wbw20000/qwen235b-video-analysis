from __future__ import annotations

import hashlib
import os
import threading
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


# ===== 全局单例：模型常驻服务 =====
_embedding_service: Optional['EmbeddingService'] = None
_embedding_service_lock = threading.Lock()


class EmbeddingService:
    """
    SigLIP 模型常驻服务（单例）

    - 模型只加载一次，常驻内存
    - 缓存 text embeddings（模板固定时直接复用）
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._select_device(config.device)

        # 加载模型（只执行一次）
        print(f"[EmbeddingService] 初始化SigLIP常驻服务...")
        self._load_model(config)

        # Text embeddings 缓存: hash(templates) -> np.ndarray
        self._text_cache: Dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()

        print(f"[EmbeddingService] 常驻服务初始化完成，模型已加载到 {self.device}")

    def _load_model(self, config: EmbeddingConfig):
        """加载SigLIP模型"""
        try:
            print(f"[EmbeddingService] 尝试从本地缓存加载模型: {config.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            print(f"[EmbeddingService] OK - Loaded from local cache")
        except Exception:
            print(f"[EmbeddingService] 本地缓存不存在，开始联网下载模型...")
            self.processor = AutoProcessor.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                config.model_name,
                trust_remote_code=True
            )
            print(f"[EmbeddingService] OK - Model downloaded and cached")

        self.model.to(self.device)
        self.model.eval()

        # 验证模型设备
        try:
            first_param = next(self.model.parameters())
            actual_device = first_param.device
            print(f"[EmbeddingService] Model actual device: {actual_device}")
            if "cuda" in str(actual_device):
                print(f"[EmbeddingService] >>> GPU acceleration ENABLED!")
        except Exception as e:
            print(f"[EmbeddingService] WARNING: Cannot verify model device: {e}")

    def _select_device(self, device_pref: str) -> str:
        """选择计算设备"""
        if device_pref == "auto":
            try:
                print(f"[EmbeddingService] 检测CUDA可用性...")
                cuda_available = torch.cuda.is_available()
                print(f"[EmbeddingService] torch.cuda.is_available() = {cuda_available}")

                if cuda_available:
                    print(f"[EmbeddingService] 选择设备: cuda")
                    return "cuda"

                print(f"[EmbeddingService] 检测MPS可用性...")
                if torch.backends.mps.is_available():
                    print(f"[EmbeddingService] 选择设备: mps")
                    return "mps"

            except Exception as e:
                # 关键修复：打印异常信息（避免特殊字符GBK编码问题）
                try:
                    print(f"[EmbeddingService] 设备检测异常: {type(e).__name__}: {str(e)}")
                except Exception:
                    print(f"[EmbeddingService] 设备检测异常（无法打印详情）")

            print(f"[EmbeddingService] 回退到CPU")
            return "cpu"

        print(f"[EmbeddingService] 使用指定设备: {device_pref}")
        return device_pref

    def _compute_cache_key(self, texts: List[str]) -> str:
        """计算模板列表的缓存key"""
        content = "|".join(sorted(texts))
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def encode_texts_cached(self, texts: List[str]) -> np.ndarray:
        """编码文本（带缓存）"""
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        cache_key = self._compute_cache_key(texts)

        with self._cache_lock:
            if cache_key in self._text_cache:
                print(f"[EmbeddingService] Text embeddings 缓存命中! (key={cache_key[:8]}..., {len(texts)} templates)")
                return self._text_cache[cache_key].copy()

        # 缓存未命中，计算embeddings
        print(f"[EmbeddingService] Text embeddings 缓存未命中，开始编码 {len(texts)} 个模板...")
        embeddings = self._encode_texts_impl(texts)

        with self._cache_lock:
            self._text_cache[cache_key] = embeddings.copy()
            print(f"[EmbeddingService] Text embeddings 已缓存 (key={cache_key[:8]}..., 当前缓存数: {len(self._text_cache)})")

        return embeddings

    def _encode_texts_impl(self, texts: List[str]) -> np.ndarray:
        """实际的文本编码逻辑"""
        outputs = []
        batch_size = max(1, self.config.batch_size)
        max_len = 64
        try:
            max_len = int(getattr(self.model.config.text_config, "max_position_embeddings", max_len))
        except Exception:
            pass
        use_amp = "cuda" in str(self.device)

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    text_features = self.model.get_text_features(**inputs)

            text_features = text_features.float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            outputs.append(text_features.cpu().numpy())

        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        """编码图像（不缓存，每次都是新图像）"""
        if not images:
            return np.zeros((0, 1), dtype=np.float32)

        outputs = []
        batch_size = max(1, self.config.batch_size)
        use_amp = "cuda" in str(self.device)

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start : start + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    image_features = self.model.get_image_features(**inputs)

            image_features = image_features.float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            outputs.append(image_features.cpu().numpy())

        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def clear_text_cache(self):
        """清空text embeddings缓存"""
        with self._cache_lock:
            self._text_cache.clear()
            print("[EmbeddingService] Text embeddings 缓存已清空")


def get_embedding_service(config: EmbeddingConfig = None) -> EmbeddingService:
    """获取EmbeddingService单例"""
    global _embedding_service

    with _embedding_service_lock:
        if _embedding_service is None:
            if config is None:
                config = EmbeddingConfig()
            _embedding_service = EmbeddingService(config)
        return _embedding_service


def cleanup_embedding_service():
    """清理EmbeddingService单例（释放GPU显存和缓存）"""
    global _embedding_service

    with _embedding_service_lock:
        if _embedding_service is not None:
            print("[EmbeddingService] 开始清理常驻服务...")

            # 清空text缓存
            _embedding_service.clear_text_cache()

            # 释放模型
            try:
                del _embedding_service.model
                del _embedding_service.processor
            except Exception as e:
                print(f"[EmbeddingService] 模型释放警告: {e}")

            # 清理CUDA缓存
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("[EmbeddingService] CUDA缓存已清理")
            except Exception as e:
                print(f"[EmbeddingService] CUDA清理警告: {e}")

            _embedding_service = None
            print("[EmbeddingService] 常驻服务已清理完成")


class EmbeddingIndexer:
    """
    SigLIP 编码与向量索引。
    - 使用 EmbeddingService 单例（模型常驻，避免重复加载）
    - 支持 faiss 内存索引，缺省回退到 numpy 余弦。
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        # 使用单例服务（模型常驻，只加载一次）
        self._service = get_embedding_service(config)
        self.device = self._service.device

        print(f"[EmbeddingIndexer] 使用常驻服务 (device={self.device})")

        self.vector_dim: Optional[int] = None
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.index = None

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        """编码图像（使用常驻服务）"""
        total_start = time.time()
        num_images = len(image_paths)
        print(f"[SigLIP] === Image Encoding START === ({num_images} images)")

        # Step 1: 磁盘 I/O + PIL 预处理
        io_start = time.time()
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
        io_time = time.time() - io_start
        print(f"[SigLIP] Step 1/2: Disk I/O: {io_time:.2f}s ({num_images} images)")

        # Step 2: 使用常驻服务编码
        gpu_start = time.time()
        embeddings = self._service.encode_images(images)
        gpu_time = time.time() - gpu_start

        total_time = time.time() - total_start
        print(f"[SigLIP] === Image Encoding END === Total: {total_time:.2f}s (I/O: {io_time:.2f}s, GPU: {gpu_time:.2f}s)")

        return embeddings

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本（使用常驻服务 + 缓存）"""
        total_start = time.time()
        num_texts = len(texts)
        print(f"[SigLIP] === Text Encoding START === ({num_texts} templates)")

        # 使用常驻服务的缓存功能
        embeddings = self._service.encode_texts_cached(texts)

        total_time = time.time() - total_start
        print(f"[SigLIP] === Text Encoding END === Total: {total_time:.2f}s ({num_texts} templates)")

        return embeddings

    def add_frame_embeddings(self, records: List[Dict]) -> List[int]:
        """
        records: [{"image_path": str, "image": PIL.Image (可选), "metadata": {...}}]
        返回添加后的内部 ID 列表。

        优化：优先使用内存图像(image字段)，避免重复磁盘I/O
        """
        if not records:
            return []

        # 收集图像：优先使用内存图像，否则从路径读取
        images = []
        from_memory = 0
        from_disk = 0
        for r in records:
            if "image" in r and r["image"] is not None:
                images.append(r["image"])
                from_memory += 1
            else:
                img = Image.open(r["image_path"]).convert("RGB")
                images.append(img)
                from_disk += 1

        print(f"[EmbeddingIndexer] 图像来源: 内存={from_memory}, 磁盘={from_disk}")

        # 直接使用PIL Image列表编码
        embeddings = self._service.encode_images(images).astype(np.float32)
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
