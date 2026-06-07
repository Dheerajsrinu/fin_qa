import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.hybrid import rrf_score
from eval import recall_at_k, mrr


def test_rrf_score_decreases_with_rank():
    assert rrf_score(1) > rrf_score(2) > rrf_score(10)


def test_rrf_score_k_parameter():
    assert rrf_score(1, k=10) > rrf_score(1, k=60)


def test_recall_at_k_hit():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["a", "b", "c"], {"d"}, k=3) == 0.0


def test_recall_at_k_respects_cutoff():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_position():
    assert abs(mrr(["a", "b", "c"], {"b"}) - 0.5) < 1e-9


def test_mrr_no_hit():
    assert mrr(["a", "b", "c"], {"d"}) == 0.0
