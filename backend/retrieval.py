"""Hybrid retrieval: BM25 keyword search + vector search, fused with RRF.

Embeddings are good at meaning ("why do cars pit?") but bad at exact tokens
like driver names and years. BM25 is the opposite. Reciprocal Rank Fusion
combines the two rankings without having to calibrate their scores.
An optional cross-encoder (F1_RERANKER=<model name>) re-scores the top hits.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional

from backend.stats_db import fold

logger = logging.getLogger(__name__)

RRF_K = 60
CANDIDATES = int(os.getenv("F1_TOP_K", "20"))
FINAL_CHUNKS = int(os.getenv("F1_FINAL_CONTEXT_CHUNKS", "4"))
STOPWORDS = set("the a an of in on and or to is are was were what who whom which how why when does did do for with by at from it its be as that this".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", fold(text)) if t not in STOPWORDS]


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused, normalized 0..1
    vector_sim: Optional[float] = None


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.lengths = [len(d) for d in docs]
        self.avg = (sum(self.lengths) / len(docs)) if docs else 0.0
        self.tf = [Counter(d) for d in docs]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, length in zip(self.tf, self.lengths):
            s = 0.0
            for term in query:
                f = tf.get(term)
                if not f:
                    continue
                s += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], collection: Any = None, embedder: Any = None) -> None:
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([tokenize(c.text + " " + str(c.metadata.get("heading", ""))) for c in chunks])
        self.collection = collection
        self.embedder = embedder
        self.reranker = None
        model = os.getenv("F1_RERANKER", "").strip()
        if model:
            try:
                from sentence_transformers import CrossEncoder

                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2" if model == "1" else model)
                logger.info("Cross-encoder reranker loaded: %s", model)
            except Exception:
                logger.exception("Could not load reranker %s; continuing without it.", model)

    @property
    def has_vectors(self) -> bool:
        return self.collection is not None and self.embedder is not None

    def _vector_ranking(self, query: str) -> list[tuple[str, float]]:
        if not self.has_vectors:
            return []
        try:
            vec = self.embedder.encode([query], normalize_embeddings=True)[0]
            res = self.collection.query(
                query_embeddings=[vec.tolist() if hasattr(vec, "tolist") else list(vec)],
                n_results=min(CANDIDATES, max(1, len(self.chunks))),
                include=["distances"],
            )
        except Exception:
            logger.exception("Vector query failed; using BM25 only.")
            return []
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [(i, max(0.0, 1.0 - float(d))) for i, d in zip(ids, dists) if i in self.by_id]

    @staticmethod
    def _metadata_bonus(query: str, chunk: Chunk) -> float:
        q = fold(query)
        meta = chunk.metadata
        bonus = 0.0
        entity = fold(str(meta.get("entity", "")))
        heading = fold(str(meta.get("heading", "")))
        if entity and len(entity) > 3 and entity in q:
            bonus += 0.15
        if heading and len(heading) > 3 and heading in q:
            bonus += 0.1
        for year in re.findall(r"\b(19[5-9]\d|20\d{2})\b", q):
            if year in str(meta.get("season", "")):
                bonus += 0.15
        return bonus

    def search(self, query: str, k: int = FINAL_CHUNKS) -> list[Hit]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        bm = self.bm25.scores(q_tokens)
        bm_rank = [(self.chunks[i].id, s) for i, s in sorted(enumerate(bm), key=lambda x: -x[1])[:CANDIDATES] if s > 0]
        vec_rank = self._vector_ranking(query)
        vec_sim = dict(vec_rank)

        fused: dict[str, float] = {}
        for ranking in (bm_rank, vec_rank):
            for rank, (cid, _) in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        if not fused:
            return []
        top = max(fused.values())
        hits = [
            Hit(self.by_id[cid], score / top + self._metadata_bonus(query, self.by_id[cid]), vec_sim.get(cid))
            for cid, score in fused.items()
        ]
        hits.sort(key=lambda h: -h.score)
        hits = hits[: max(k * 3, 10)]

        if self.reranker is not None and hits:
            try:
                ce = self.reranker.predict([(query, h.chunk.text) for h in hits])
                for h, s in zip(hits, ce):
                    h.score = 1 / (1 + math.exp(-float(s)))
                hits.sort(key=lambda h: -h.score)
            except Exception:
                logger.exception("Reranker failed; keeping fused order.")

        return hits[:k]
