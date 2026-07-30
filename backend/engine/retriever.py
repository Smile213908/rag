"""bge-m3 检索器 + Chroma 向量库:稠密向量持久化 + 混合召回。

架构(见大脑笔记「向量数据库」「混合检索与 Rerank」):
- bge-m3(sentence-transformers 加载)负责**生成稠密向量**(一次性,CPU 上最贵)。
- Chroma(docker 容器)负责**持久化存储 + 稠密检索 + 元数据过滤**——重启不重算。
- 混合检索第二路用 **BM25**(rank_bm25,纯 Python、CPU 友好)补关键词精确匹配,
  弥补稠密向量对专有名词/编号的盲区。
- 两路融合用**加权 RRF**(倒数排名融合):稠密余弦相似度与 BM25 分数量纲不同,
  直接加权会被 min-max 归一化失真(BM25 全员低分时最高分仍被放大为 1),
  RRF 只依赖名次,对量纲免疫。

索引建一次落盘 Chroma,下次启动直接复用,跳过 bge-m3 编码;BM25 索引按
Chroma 集合里的 build_id 缓存到本地 .cache/,命中即跳过全量回读重建。
"""
from __future__ import annotations

import os
import re

import numpy as np

from engine.chunking import Chunk
from engine.paths import CACHE_DIR, MODEL_DIR

# repo id → 本地模型目录名
_MODEL_DIR_MAP = {
    "BAAI/bge-m3": "bge-m3",
    "BAAI/bge-reranker-base": "bge-reranker-base",
}


def _local_model_path(repo_id: str) -> str:
    """把 HuggingFace repo id 解析为本地模型路径(model/<name>/snapshots/master)。

    模型已迁移到项目 model/ 目录,直接加载、不再联网下载。
    """
    dir_name = _MODEL_DIR_MAP.get(repo_id, repo_id.replace("/", "--"))
    path = os.path.join(MODEL_DIR, dir_name, "snapshots", "master")
    if not os.path.exists(path):
        # 本地不存在才回退到 modelscope 下载(首次/模型缺失时)
        from modelscope import snapshot_download
        path = snapshot_download(repo_id)
    return path


# BM25 索引本地缓存目录(CACHE_DIR,按 Chroma 集合里的 build_id 校验有效性)

# RRF(倒数排名融合)常数,经验取值 60
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """中文友好的简易分词:按 2-gram 切分(无需 jieba 依赖,效果够做 BM25)。"""
    text = re.sub(r"\s+", "", text.lower())
    return [text[i:i + 2] for i in range(len(text) - 1)] or [text]


class BGERetriever:
    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        collection: str = "rag_docs",
    ):
        import chromadb
        self.client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        # cosine 空间:配合归一化向量,距离=1-余弦相似度
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        # 嵌入模型来源优先级:显式传入 > 注册表当前生效 > 默认 bge-m3
        path = model_name or self._active_model_path("embedding") \
            or _local_model_path("BAAI/bge-m3")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(path, device=device)

        self.chunks: list[Chunk] = []
        self._bm25 = None  # BM25 索引(内存)

    @staticmethod
    def _active_model_path(model_type: str) -> str | None:
        """从模型注册表读当前生效模型的路径(无注册表时返回 None)。"""
        try:
            from engine.model_registry import ModelRegistry
            rec = ModelRegistry().get_active(model_type)
            return rec.path if rec else None
        except Exception:
            return None

    # ---------- 索引 ----------
    def index(self, chunks: list[Chunk], batch_size: int = 16,
              overwrite: bool = True) -> None:
        if overwrite and self.col.count() > 0:
            self.client.delete_collection(self.col.name)
            self.col = self.client.get_or_create_collection(
                name=self.col.name, metadata={"hnsw:space": "cosine"}
            )
        self.chunks = chunks
        self._encode_upsert(chunks, batch_size)
        self._after_mutation()

    def _encode_upsert(self, chunks: list[Chunk], batch_size: int = 16) -> None:
        """编码并写入 Chroma(upsert,幂等;同 cid 覆盖)。"""
        texts = [c.text for c in chunks]
        # 稠密向量(bge-m3),归一化后进 Chroma
        dense = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        self.col.upsert(
            ids=[c.cid for c in chunks],
            embeddings=dense.tolist(),
            documents=texts,
            metadatas=[{"doc_id": c.doc_id, **c.meta} for c in chunks],
        )

    def _after_mutation(self) -> None:
        """集合内容变更后的统一收尾:刷新 build_id 指纹 + 重建 BM25 + 刷缓存。

        注意:Chroma modify 校验拒绝元数据中出现 hnsw:space(距离函数创建后
        不可改),这里只写 build_id,实际距离配置不受影响。
        """
        import uuid
        build_id = uuid.uuid4().hex
        self.col.modify(metadata={"build_id": build_id})
        self._build_bm25()
        self._save_bm25_cache(build_id)

    # ---------- 增量维护(知识库管理接口用) ----------
    def add_chunks(self, chunks: list[Chunk], batch_size: int = 16) -> None:
        """增量入库:不清空既有集合,追加编码 + upsert,随后重建 BM25 刷缓存。"""
        if not chunks:
            return
        self._encode_upsert(chunks, batch_size)
        known = {c.cid for c in self.chunks}
        self.chunks.extend(c for c in chunks if c.cid not in known)
        self._after_mutation()

    def remove_doc(self, doc_id: str) -> int:
        """按 doc_id 级联删除(向量 + 文档 + BM25 项),返回删除块数。"""
        removed = sum(1 for c in self.chunks if c.doc_id == doc_id)
        if removed:
            self.col.delete(where={"doc_id": doc_id})
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            self._after_mutation()
        return removed

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    # ---------- BM25 缓存(避免每次启动全量回读 Chroma + 重分词重建) ----------
    def _bm25_cache_path(self) -> str:
        return os.path.join(CACHE_DIR, f"bm25_{self.col.name}.pkl")

    def _save_bm25_cache(self, build_id: str) -> None:
        import pickle
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self._bm25_cache_path(), "wb") as f:
            pickle.dump({"build_id": build_id,
                         "chunks": self.chunks, "bm25": self._bm25}, f)

    def _load_bm25_cache(self, build_id: str | None) -> bool:
        """缓存存在且 build_id 与 Chroma 集合一致时,恢复 chunks + BM25。"""
        if not build_id:
            return False
        import pickle
        path = self._bm25_cache_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)  # 本进程自产的本地缓存,来源可信
        except Exception:
            return False
        if data.get("build_id") != build_id:
            return False
        self.chunks, self._bm25 = data["chunks"], data["bm25"]
        return True

    def load_existing(self) -> bool:
        """Chroma 已有索引时恢复块信息与 BM25(跳过 bge 编码)。返回是否有数据。

        优先命中本地 BM25 缓存;缓存缺失/失效才全量回读 Chroma 重建,并刷新缓存。
        """
        if self.col.count() == 0:
            return False
        build_id = (self.col.metadata or {}).get("build_id")
        if self._load_bm25_cache(build_id):
            return True
        got = self.col.get(include=["documents", "metadatas"])
        self.chunks = [
            Chunk(text=doc, doc_id=meta.get("doc_id", ""),
                  chunk_id=cid.split("#")[-1], meta=meta)
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]
        self._build_bm25()
        if not build_id:  # 旧版无 build_id 的集合,补一个完成自愈
            import uuid
            build_id = uuid.uuid4().hex
            # 过滤 hnsw:* 键:Chroma modify 校验拒绝元数据中出现 hnsw:space
            base = {k: v for k, v in (self.col.metadata or {}).items()
                    if not k.startswith("hnsw:")}
            self.col.modify(metadata={**base, "build_id": build_id})
        self._save_bm25_cache(build_id)
        return True

    # ---------- 检索 ----------
    def search(self, query: str, top_k: int = 50, dense_weight: float = 0.7,
               where: dict | None = None) -> list[tuple[Chunk, float]]:
        """混合检索:Chroma 稠密 + BM25,加权 RRF(倒数排名)融合。

        按名次而非原始分数融合:稠密余弦相似度与 BM25 分数量纲不同,直接加权
        会被 BM25 的 min-max 归一化失真(全员低分时最高分仍被放大为 1);
        RRF 对量纲免疫,dense_weight 退化为两路的权重。where 为 Chroma 元数据过滤。
        """
        if not self.chunks:
            raise RuntimeError("索引为空,先 index() 或 load_existing()")

        cid_to_idx = {c.cid: i for i, c in enumerate(self.chunks)}
        fused = np.zeros(len(self.chunks), dtype=np.float64)

        # 稠密路:bge-m3 编码问题 → Chroma 检索(返回结果已按距离升序排名)
        q_dense = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)[0]
        res = self.col.query(
            query_embeddings=[q_dense.tolist()],
            n_results=min(top_k, self.col.count()),
            where=where, include=["distances"],
        )
        for rank, cid in enumerate(res["ids"][0]):
            idx = cid_to_idx.get(cid)
            if idx is not None:
                fused[idx] += dense_weight / (_RRF_K + rank + 1)

        # 稀疏路:BM25 按分数降序得名次(0 分无匹配,不计名次)
        if self._bm25 is not None:
            bm = np.asarray(self._bm25.get_scores(_tokenize(query)),
                            dtype=np.float64)
            for rank, idx in enumerate(np.argsort(-bm)):
                if bm[idx] <= 0:
                    break
                fused[idx] += (1 - dense_weight) / (_RRF_K + rank + 1)

        top_idx = np.argsort(-fused)[:top_k]
        return [(self.chunks[i], float(fused[i])) for i in top_idx if fused[i] > 0]
