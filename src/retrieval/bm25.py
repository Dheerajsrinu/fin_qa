import pickle
from pathlib import Path

import bm25s
import numpy as np

import config
from index.bm25_indexer import BM25Indexer, tokenize


class BM25Retriever:
    def __init__(self, indexer: BM25Indexer, corpus: dict):
        self.indexer = indexer
        self.corpus = corpus

    @classmethod
    def load(
        cls,
        index_path: Path = config.BM25_INDEX_PATH,
        corpus_path: Path = config.CORPUS_CACHE_PATH,
    ) -> "BM25Retriever":
        with open(index_path, "rb") as f:
            indexer = pickle.load(f)
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)
        return cls(indexer, corpus)

    def search(self, query: str, top_k: int = config.TOP_K) -> list[tuple[str, float, str]]:
        k = min(top_k, len(self.indexer.doc_ids))
        query_tokens = bm25s.tokenize([query], show_progress=False)
        indices, scores = self.indexer.bm25.retrieve(query_tokens, k=k, show_progress=False)

        # indices and scores are shape (1, k) — squeeze the batch dimension
        return [
            (self.indexer.doc_ids[i], float(s), self.corpus[self.indexer.doc_ids[i]]["text"])
            for i, s in zip(indices[0], scores[0])
        ]
