from __future__ import annotations

import os
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
        self.processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

        self.vector_dim: Optional[int] = None
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.index = None

    def _select_device(self, device_pref: str) -> str:
        if device_pref == "auto":
            try:
                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():  # type: ignore
                    return "mps"
            except Exception:
                pass
            return "cpu"
        return device_pref

    def encode_images(self, image_paths: List[str]) -> np.ndarray:
        batches = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            batches.append(img)

        outputs = []
        batch_size = max(1, self.config.batch_size)
        for start in range(0, len(batches), batch_size):
            batch_imgs = batches[start : start + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            outputs.append(image_features.cpu().numpy())

        return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float32)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        outputs = []
        batch_size = max(1, self.config.batch_size)
        max_len = 64
        try:
            max_len = int(getattr(self.model.config.text_config, "max_position_embeddings", max_len))
        except Exception:
            pass
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            outputs.append(text_features.cpu().numpy())
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
