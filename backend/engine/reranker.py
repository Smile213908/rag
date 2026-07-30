"""bge-reranker-base 精排:对召回的小集合成对打分重排。

原理(见大脑笔记「混合检索与 Rerank」):
Cross-Encoder 把「问题+候选块」**成对**输入模型直接打相关性分,
精度高于向量检索的「相似」,但慢——只适合对召回的 Top-N 精排,不适合全库。
这里用 sentence-transformers 的 CrossEncoder 加载 bge-reranker-base(稳定)。
"""
from __future__ import annotations

import os

from engine.chunking import Chunk
from engine.paths import MODEL_DIR

_MODEL_DIR_MAP = {
    "BAAI/bge-m3": "bge-m3",
    "BAAI/bge-reranker-base": "bge-reranker-base",
}


def _local_model_path(repo_id: str) -> str:
    """解析为本地 model/<name>/snapshots/master;不存在才回退 modelscope 下载。"""
    dir_name = _MODEL_DIR_MAP.get(repo_id, repo_id.replace("/", "--"))
    path = os.path.join(MODEL_DIR, dir_name, "snapshots", "master")
    if not os.path.exists(path):
        from modelscope import snapshot_download
        path = snapshot_download(repo_id)
    return path


class BGEReranker:
    def __init__(self, model_name: str | None = None, device: str = "cpu"):
        from sentence_transformers import CrossEncoder
        # 重排模型来源:显式传入 > 注册表当前生效 > 默认 bge-reranker-base
        path = model_name or self._active_model_path("reranker") \
            or _local_model_path("BAAI/bge-reranker-base")
        self.model = CrossEncoder(path, device=device)

    @staticmethod
    def _active_model_path(model_type: str) -> str | None:
        try:
            from engine.model_registry import ModelRegistry
            rec = ModelRegistry().get_active(model_type)
            return rec.path if rec else None
        except Exception:
            return None

    def rerank(self, query: str, candidates: list[Chunk],
               top_k: int = 5) -> list[tuple[Chunk, float]]:
        """对候选块精排,返回 [(chunk, score)] 按分数降序的前 top_k 个。"""
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(c, float(s)) for c, s in ranked[:top_k]]
