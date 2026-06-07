import pickle

import config
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever


def rrf_score(rank: int, k: int = config.RRF_K) -> float:
    return 1.0 / (k + rank)


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a score list to [0, 1]."""
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever):
        self.bm25  = bm25
        self.dense = dense

    @classmethod
    def load(cls) -> "HybridRetriever":
        with open(config.CORPUS_CACHE_PATH, "rb") as f:
            corpus = pickle.load(f)
        return cls(
            BM25Retriever.load(corpus=corpus),
            DenseRetriever.load(corpus=corpus),
        )

    def search(
        self,
        query: str,
        top_k: int  = config.TOP_K,
        method: str = "linear",
        alpha: float = 0.3,
        rrf_k: int  = config.RRF_K,
    ) -> list[tuple[str, float, str]]:
        bm25_results  = self.bm25.search(query,  top_k=config.CANDIDATE_K)
        dense_results = self.dense.search(query, top_k=config.CANDIDATE_K)

        if method == "rrf":
            return self._rrf(bm25_results, dense_results, top_k, rrf_k)
        if method == "linear":
            return self._linear(bm25_results, dense_results, top_k, alpha)
        raise ValueError(f"Unknown fusion method: {method!r}. Use 'rrf' or 'linear'.")

    def _rrf(self, bm25_res, dense_res, top_k, rrf_k):
        scores: dict[str, float] = {}
        texts:  dict[str, str]   = {}

        for rank, (doc_id, _, text) in enumerate(bm25_res, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank, rrf_k)
            texts[doc_id]  = text

        for rank, (doc_id, _, text) in enumerate(dense_res, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + rrf_score(rank, rrf_k)
            texts[doc_id]  = text

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, texts[doc_id]) for doc_id, score in ranked]

    def _linear(self, bm25_res, dense_res, top_k, alpha):
        bm25_norm  = _normalize([s for _, s, _ in bm25_res])
        dense_norm = _normalize([s for _, s, _ in dense_res])

        bm25_map  = {doc_id: (ns, text) for (doc_id, _, text), ns in zip(bm25_res,  bm25_norm)}
        dense_map = {doc_id: (ns, text) for (doc_id, _, text), ns in zip(dense_res, dense_norm)}

        scores: dict[str, float] = {}
        texts:  dict[str, str]   = {}

        for doc_id in set(bm25_map) | set(dense_map):
            b_score, b_text = bm25_map.get(doc_id,  (0.0, ""))
            d_score, d_text = dense_map.get(doc_id, (0.0, ""))
            scores[doc_id] = alpha * b_score + (1 - alpha) * d_score
            texts[doc_id]  = b_text or d_text

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, texts[doc_id]) for doc_id, score in ranked]
