"""
accident_rag.py - Image RAG for accident exemplar retrieval

Phase B: Unleashing VLMs (arXiv:2601.10551) visual exemplar injection.
Retrieves confirmed-TP accident keyframes by camera type and prepends them
to the VLM context as visual reference samples.

Usage:
    rag = AccidentRAGDatabase(exemplars_dir="data/accident_exemplars")
    frames = rag.retrieve(cam_type="roadside", top_k=2)
    # frames: List[str] — absolute paths to selected exemplar JPGs
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional


_CAM_TYPE_DIRS = {
    "roadside":   "roadside",
    "elec_police": "elec_police",
    "unknown":    "general",
}


class AccidentRAGDatabase:
    """
    Lightweight image-based RAG for accident exemplars.

    Directory layout expected:
        exemplars_dir/
          roadside/      ← RoadsideCamera confirmed-TP frames
          elec_police/   ← 电警摄像机 confirmed-TP frames
          general/       ← other camera types

    Each category contains pre-selected JPG frames extracted from confirmed
    true-positive accident videos.  The retrieval strategy is random sampling
    (with a fixed seed per call for reproducibility), which is sufficient
    because all stored frames are already curated positives.
    """

    def __init__(self, exemplars_dir: str = "data/accident_exemplars"):
        self.base = Path(exemplars_dir)
        self._index: dict[str, List[Path]] = {}
        self._load()

    def _load(self):
        for cam_type, subdir in _CAM_TYPE_DIRS.items():
            folder = self.base / subdir
            if folder.is_dir():
                frames = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
                self._index[cam_type] = frames
            else:
                self._index[cam_type] = []

        total = sum(len(v) for v in self._index.values())
        if total == 0:
            import warnings
            warnings.warn(
                f"[AccidentRAG] No exemplar frames found in {self.base}. "
                "Image RAG will be skipped.",
                stacklevel=2,
            )

    def is_ready(self) -> bool:
        return any(len(v) > 0 for v in self._index.values())

    def retrieve(
        self,
        cam_type: str = "unknown",
        top_k: int = 2,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Return up to top_k exemplar frame paths for the given camera type.

        Falls back to "unknown" (general) pool when the specific category
        has fewer than top_k frames.
        """
        pool = list(self._index.get(cam_type, []))

        # Fallback: supplement from general pool if needed
        if len(pool) < top_k:
            general = self._index.get("unknown", [])
            pool = list(set(pool + general))

        if not pool:
            return []

        rng = random.Random(seed)
        selected = rng.sample(pool, min(top_k, len(pool)))
        return [str(p) for p in selected]

    def stats(self) -> dict:
        return {k: len(v) for k, v in self._index.items()}
