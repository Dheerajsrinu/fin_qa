"""
BM25 indexer and retriever tests.

What this file tests and why:

1. tokenize()
   - Core preprocessing step. Every doc and query goes through this.
   - Must lowercase, split on word boundaries, strip punctuation.

2. BM25Indexer.build()
   - Verifies the index stores the right number of doc IDs.
   - Verifies BM25 object is created (not None).

3. BM25Retriever.search() — result count
   - top_k=3 must return exactly 3 results, not more, not less.

4. BM25Retriever.search() — result structure
   - Each result must be (doc_id: str, score: float, text: str).

5. BM25Retriever.search() — score ordering
   - Results must be sorted descending by score.
   - A query with a keyword match must score higher than one without.

6. BM25Retriever.search() — relevance signal
   - A passage containing the query term must rank above one that doesn't.
   - This is the fundamental correctness check for BM25.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from index.bm25_indexer import BM25Indexer, tokenize


# ---------- tokenize() ----------

def test_tokenize_lowercases():
    assert tokenize("Short SELLING") == ["short", "selling"]


def test_tokenize_strips_punctuation():
    assert tokenize("what's the price?") == ["what", "s", "the", "price"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_tokenize_numbers():
    assert "401k" in tokenize("401k retirement plan")


# ---------- BM25Indexer ----------

MINI_CORPUS = {
    "d1": {"title": "", "text": "short selling means borrowing shares"},
    "d2": {"title": "", "text": "capital gains tax on stock sales"},
    "d3": {"title": "", "text": "inflation reduces purchasing power of money"},
}


@pytest.fixture
def built_indexer():
    idx = BM25Indexer()
    idx.build(MINI_CORPUS)
    return idx


def test_indexer_doc_count(built_indexer):
    assert len(built_indexer.doc_ids) == len(MINI_CORPUS)


def test_indexer_bm25_not_none(built_indexer):
    assert built_indexer.bm25 is not None


def test_indexer_doc_ids_order(built_indexer):
    assert set(built_indexer.doc_ids) == set(MINI_CORPUS.keys())


# ---------- BM25Retriever (using mini corpus, no disk I/O) ----------

@pytest.fixture
def retriever(built_indexer):
    from retrieval.bm25 import BM25Retriever
    return BM25Retriever(built_indexer, MINI_CORPUS)


def test_search_returns_top_k(retriever):
    results = retriever.search("short selling", top_k=2)
    assert len(results) == 2


def test_search_result_structure(retriever):
    results = retriever.search("capital gains", top_k=1)
    doc_id, score, text = results[0]
    assert isinstance(doc_id, str)
    assert isinstance(score, float)
    assert isinstance(text, str)


def test_search_scores_descending(retriever):
    results = retriever.search("short selling stock", top_k=3)
    scores = [s for _, s, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_search_keyword_relevance(retriever):
    results = retriever.search("short selling", top_k=3)
    top_doc_id = results[0][0]
    assert top_doc_id == "d1", f"Expected d1 (about short selling), got {top_doc_id}"


def test_search_top_k_larger_than_corpus(retriever):
    results = retriever.search("money", top_k=100)
    assert len(results) == len(MINI_CORPUS)
