"""
Hybrid retriever tests.

We use lightweight stub retrievers (no disk I/O, no models) so every test
runs in milliseconds.

What this file tests and why:

1. rrf_score()
   - Decreases as rank increases (better rank = higher score).
   - Larger k = smaller score for same rank (k dampens rank differences).

2. _normalize()
   - Maps min to 0, max to 1.
   - Handles all-equal list without division-by-zero.

3. HybridRetriever._rrf()
   - A doc that appears in BOTH lists scores higher than one in only one list.
   - The top result is actually the highest-scoring doc after fusion.
   - Returns exactly top_k results.

4. HybridRetriever._linear()
   - Same structural checks as RRF.
   - alpha=1.0 should rank purely by BM25.
   - alpha=0.0 should rank purely by dense.

5. HybridRetriever.search() routing
   - method='rrf' and method='linear' both work.
   - Unknown method raises ValueError.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.hybrid import HybridRetriever, _normalize, rrf_score


# ── helpers ──────────────────────────────────────────────────────────────────

def make_retriever(bm25_results, dense_results):
    """Stub retriever that returns fixed results regardless of query."""

    class _Stub:
        def __init__(self, results):
            self._results = results

        def search(self, query, top_k=10):
            return self._results[:top_k]

    return HybridRetriever(_Stub(bm25_results), _Stub(dense_results))


BM25_RESULTS = [
    ("d1", 10.0, "doc one"),
    ("d2",  8.0, "doc two"),
    ("d3",  6.0, "doc three"),
    ("d4",  4.0, "doc four"),
    ("d5",  2.0, "doc five"),
]

DENSE_RESULTS = [
    ("d3", 0.95, "doc three"),
    ("d1", 0.90, "doc one"),
    ("d6", 0.85, "doc six"),
    ("d7", 0.80, "doc seven"),
    ("d2", 0.75, "doc two"),
]


# ── rrf_score ────────────────────────────────────────────────────────────────

def test_rrf_score_decreases_with_rank():
    assert rrf_score(1) > rrf_score(2) > rrf_score(10)


def test_rrf_score_larger_k_smaller_score():
    assert rrf_score(1, k=10) > rrf_score(1, k=60)


def test_rrf_score_positive():
    assert rrf_score(100) > 0


# ── _normalize ───────────────────────────────────────────────────────────────

def test_normalize_min_is_zero():
    result = _normalize([2.0, 5.0, 8.0])
    assert result[0] == pytest.approx(0.0)


def test_normalize_max_is_one():
    result = _normalize([2.0, 5.0, 8.0])
    assert result[-1] == pytest.approx(1.0)


def test_normalize_all_equal():
    result = _normalize([3.0, 3.0, 3.0])
    assert result == [1.0, 1.0, 1.0]


# ── RRF fusion ───────────────────────────────────────────────────────────────

def test_rrf_top_k_count():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    assert len(r.search("q", top_k=3, method="rrf")) == 3


def test_rrf_result_structure():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    doc_id, score, text = r.search("q", top_k=1, method="rrf")[0]
    assert isinstance(doc_id, str)
    assert isinstance(score, float)
    assert isinstance(text, str)


def test_rrf_scores_descending():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    scores = [s for _, s, _ in r.search("q", top_k=5, method="rrf")]
    assert scores == sorted(scores, reverse=True)


def test_rrf_doc_in_both_lists_ranks_higher():
    # d1 and d3 appear in both BM25 and dense — should beat d6 (dense only)
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    results = r.search("q", top_k=6, method="rrf")
    top_ids = [doc_id for doc_id, _, _ in results[:4]]
    assert "d1" in top_ids
    assert "d3" in top_ids


# ── Linear fusion ─────────────────────────────────────────────────────────────

def test_linear_top_k_count():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    assert len(r.search("q", top_k=3, method="linear", alpha=0.5)) == 3


def test_linear_scores_descending():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    scores = [s for _, s, _ in r.search("q", top_k=5, method="linear", alpha=0.5)]
    assert scores == sorted(scores, reverse=True)


def test_linear_alpha_1_follows_bm25_order():
    # alpha=1.0 means weight is 100% BM25 — top result must be d1 (BM25 rank 1)
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    top_id = r.search("q", top_k=1, method="linear", alpha=1.0)[0][0]
    assert top_id == "d1"


def test_linear_alpha_0_follows_dense_order():
    # alpha=0.0 means weight is 100% dense — top result must be d3 (dense rank 1)
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    top_id = r.search("q", top_k=1, method="linear", alpha=0.0)[0][0]
    assert top_id == "d3"


# ── routing ───────────────────────────────────────────────────────────────────

def test_search_unknown_method_raises():
    r = make_retriever(BM25_RESULTS, DENSE_RESULTS)
    with pytest.raises(ValueError):
        r.search("q", method="unknown")
