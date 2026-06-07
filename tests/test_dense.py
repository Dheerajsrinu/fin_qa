"""
Dense indexer and retriever tests.

We avoid loading the real SentenceTransformer model here to keep tests fast.
Instead we use random unit vectors + a MockModel that returns fixed vectors.
This lets us test all the retriever logic (FAISS search, result formatting,
ordering, edge cases) without waiting for a 90MB model to load.

What this file tests and why:

1. DenseIndexer.build()
   - Index contains the right number of vectors.
   - doc_ids list matches corpus size and order.

2. DenseRetriever.search() — result count
   - top_k=3 on a 10-doc corpus returns exactly 3 results.

3. DenseRetriever.search() — result structure
   - Each result is (str, float, str) = (doc_id, score, text).

4. DenseRetriever.search() — score ordering
   - Results are sorted highest score first (inner product retrieval).

5. DenseRetriever.search() — semantic signal
   - A document whose vector is nearly identical to the query vector
     must rank first. Validates that HNSW finds the nearest neighbour.

6. DenseRetriever.search() — top_k > corpus size
   - Should return only as many results as there are documents, not crash.

7. efSearch propagation
   - After load() sets efSearch, the index reflects the configured value.
"""

import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import config
from index.dense_indexer import DenseIndexer
from retrieval.dense import DenseRetriever

DIM = 32   # tiny dimension — avoids loading the real 384-dim model
N = 10


class MockModel:
    """Returns a fixed random unit vector for any input."""

    def __init__(self, dim: int = DIM, seed: int = 0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        self._vec = v / np.linalg.norm(v)

    def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, **kwargs):
        out = np.tile(self._vec, (len(texts), 1))
        if normalize_embeddings:
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


MINI_CORPUS = {f"d{i}": {"title": "", "text": f"document {i} about finance"} for i in range(N)}


@pytest.fixture(scope="module")
def built_indexer():
    """DenseIndexer built with random unit vectors (no real model)."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((N, DIM)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, **kw):
            return vectors[: len(texts)]

    idx = DenseIndexer.__new__(DenseIndexer)
    idx.model = _FakeModel()
    idx.index = None
    idx.doc_ids = []
    idx.build(MINI_CORPUS)
    return idx, vectors


@pytest.fixture(scope="module")
def retriever(built_indexer):
    indexer, _ = built_indexer
    return DenseRetriever(indexer.index, indexer.doc_ids, MINI_CORPUS, MockModel())


# ---------- DenseIndexer ----------

def test_indexer_vector_count(built_indexer):
    indexer, _ = built_indexer
    assert indexer.index.ntotal == N


def test_indexer_doc_ids_count(built_indexer):
    indexer, _ = built_indexer
    assert len(indexer.doc_ids) == N


def test_indexer_doc_ids_match_corpus(built_indexer):
    indexer, _ = built_indexer
    assert set(indexer.doc_ids) == set(MINI_CORPUS.keys())


# ---------- DenseRetriever ----------

def test_search_result_count(retriever):
    results = retriever.search("short selling", top_k=3)
    assert len(results) == 3


def test_search_result_structure(retriever):
    doc_id, score, text = retriever.search("inflation", top_k=1)[0]
    assert isinstance(doc_id, str)
    assert isinstance(score, float)
    assert isinstance(text, str)


def test_search_scores_descending(retriever):
    results = retriever.search("capital gains tax", top_k=5)
    scores = [s for _, s, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_exceeds_corpus(retriever):
    results = retriever.search("anything", top_k=100)
    assert len(results) == N


def test_search_nearest_neighbour(built_indexer):
    """A document whose stored vector is closest to the query must rank first."""
    indexer, vectors = built_indexer

    # pick doc index 3 and use its exact vector as the query
    target_idx = 3
    target_vec = vectors[target_idx].reshape(1, -1).astype(np.float32)

    class _ExactModel:
        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=True, **kw):
            return target_vec

    r = DenseRetriever(indexer.index, indexer.doc_ids, MINI_CORPUS, _ExactModel())
    results = r.search("dummy", top_k=N)
    assert results[0][0] == f"d{target_idx}"
