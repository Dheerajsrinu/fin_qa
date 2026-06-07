"""
Evaluation harness — runs every dev query through each retriever and reports the
metrics the assignment grades on.

Metrics:
  - Recall@10 : is a gold passage anywhere in the top 10? (primary metric)
  - MRR       : 1/rank of the first relevant result (rewards ranking it high)
  - Latency   : cold (first 20 q) and warm (after 100-q warmup) p50/p95
  - Peak RAM  : process RSS after the retriever + index are loaded
  - Stratified: the above split by query length (short vs long), because a good
                average can hide a subpopulation the system silently fails on.

Run:
    PYTHONPATH=src python src/eval.py                  # all three retrievers
    PYTHONPATH=src python src/eval.py --retriever dense
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent))

from beir.datasets.data_loader import GenericDataLoader
import config
from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever


# ── core metrics (unit-tested) ────────────────────────────────────────────────

def recall_at_k(results: list[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant doc is in the top-k results, else 0.0.

    FiQA has ~1 gold passage per query, so binary hit/miss @k is the meaningful
    form of recall here (a fractional |hits|/|relevant| would almost always be
    0 or 1 anyway).
    """
    return 1.0 if relevant & set(results[:k]) else 0.0


def mrr(results: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result (0.0 if none found)."""
    for rank, doc_id in enumerate(results, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


# ── latency ───────────────────────────────────────────────────────────────────

def measure_latency(
    retriever,
    queries: list[str],
    top_k: int = config.TOP_K,
    search_kwargs: dict | None = None,
) -> dict:
    """Cold vs warm p50/p95, in milliseconds.

    cold   = first 20 queries (model/index just loaded, caches cold)
    warm   = 100 queries measured after a 100-query warmup
    The cold/warm split is itself a graded deliverable — see DESIGN cold-vs-warm.
    """
    search_kwargs = search_kwargs or {}

    def timed(q):
        t0 = time.perf_counter()
        retriever.search(q, top_k=top_k, **search_kwargs)
        return (time.perf_counter() - t0) * 1000

    n = len(queries)
    cold_n = min(20, n)
    cold = [timed(q) for q in queries[:cold_n]]

    # warmup (not measured) then a measured warm window
    for q in queries[cold_n:cold_n + 100]:
        retriever.search(q, top_k=top_k, **search_kwargs)
    warm_slice = queries[cold_n + 100:cold_n + 200] or queries[:cold_n]
    warm = [timed(q) for q in warm_slice]

    return {
        "cold_p50": float(np.percentile(cold, 50)),
        "cold_p95": float(np.percentile(cold, 95)),
        "warm_p50": float(np.percentile(warm, 50)),
        "warm_p95": float(np.percentile(warm, 95)),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def peak_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _quality(retriever, query_items, qrels, top_k, search_kwargs):
    """Recall@10 and MRR averaged over the given (qid, text) items."""
    recalls, mrrs = [], []
    for qid, text in query_items:
        results = retriever.search(text, top_k=top_k, **search_kwargs)
        retrieved = [doc_id for doc_id, _, _ in results]
        relevant = set(qrels.get(qid, {}).keys())
        recalls.append(recall_at_k(retrieved, relevant, k=top_k))
        mrrs.append(mrr(retrieved, relevant))
    return {
        "n": len(query_items),
        f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def _bucket_by_query_length(query_items):
    """Bucket (qid, text) by token count: short <5, medium 5-15, long >15.
    (Token count approximated by whitespace word count.)"""
    buckets = {"short_lt5": [], "medium_5_15": [], "long_gt15": []}
    for it in query_items:
        n = len(it[1].split())
        if n < 5:
            buckets["short_lt5"].append(it)
        elif n <= 15:
            buckets["medium_5_15"].append(it)
        else:
            buckets["long_gt15"].append(it)
    return buckets


def _split_by_gold_doc_length(query_items, qrels, corpus):
    """Split queries by whether their gold passage is among the top 10% longest
    docs in the corpus. Tests whether the system is biased toward short/long docs."""
    doc_len = {d: len((v.get("title", "") + " " + v.get("text", "")).split())
               for d, v in corpus.items()}
    threshold = float(np.percentile(list(doc_len.values()), 90))
    long_docs = {d for d, n in doc_len.items() if n >= threshold}

    gold_long, rest = [], []
    for qid, text in query_items:
        gold_ids = set(qrels.get(qid, {}).keys())
        (gold_long if gold_ids & long_docs else rest).append((qid, text))
    return threshold, gold_long, rest


# ── full evaluation of one retriever ──────────────────────────────────────────

def evaluate(
    retriever,
    queries: dict,
    qrels: dict,
    corpus: dict,
    top_k: int = config.TOP_K,
    search_kwargs: dict | None = None,
) -> dict:
    search_kwargs = search_kwargs or {}
    query_items = list(queries.items())                 # [(qid, text), ...]
    query_texts = [text for _, text in query_items]

    # Measure latency FIRST, right after load. The quality sweep below runs
    # ~1500 searches and saturates the CPU; measuring latency after it would
    # capture thermal throttling rather than steady-state serving. Cold latency
    # here is a genuine cold start; warm follows its own 100-query warmup.
    latency = measure_latency(retriever, query_texts, top_k, search_kwargs)

    overall = _quality(retriever, query_items, qrels, top_k, search_kwargs)

    qlen_buckets = _bucket_by_query_length(query_items)
    by_query_length = {
        name: _quality(retriever, items, qrels, top_k, search_kwargs)
        for name, items in qlen_buckets.items()
    }

    gold_thr, gold_long, rest = _split_by_gold_doc_length(query_items, qrels, corpus)
    by_gold_doc_length = {
        "long_doc_threshold_words": gold_thr,
        "gold_in_top10pct_longest": _quality(retriever, gold_long, qrels, top_k, search_kwargs),
        "rest":                     _quality(retriever, rest, qrels, top_k, search_kwargs),
    }

    stratified = {
        "by_query_length": by_query_length,
        "by_gold_doc_length": by_gold_doc_length,
    }

    return {
        **overall,
        "latency_ms": latency,
        "passes_latency_50ms": latency["warm_p95"] <= 50,
        "stratified": stratified,
        "search_config": search_kwargs or {"method": "default"},
    }


# ── retriever registry ────────────────────────────────────────────────────────
# Hybrid is evaluated at its chosen operating point: linear fusion, alpha=0.3
# (best Recall@10 within the 50ms budget once ONNX encode is in place).

def _load_retriever(name: str):
    if name == "bm25":
        return BM25Retriever.load(), {}
    if name == "dense":
        return DenseRetriever.load(), {}
    if name == "hybrid":
        return HybridRetriever.load(), {"method": "linear", "alpha": 0.3}
    raise ValueError(f"unknown retriever {name!r}")


def _benchmark_one(name: str, queries: dict, qrels: dict, corpus: dict) -> dict:
    """Load one retriever and evaluate it, measuring its RAM as a delta from the
    pre-load baseline. Meant to run in its own process (see main) so the index
    RAM and the latency aren't contaminated by other retrievers' footprint or by
    thermal throttling from a preceding heavy sweep."""
    gc.collect()
    base_rss = peak_ram_mb()                       # ML stack already imported
    retriever, search_kwargs = _load_retriever(name)
    after_rss = peak_ram_mb()

    result = evaluate(retriever, queries, qrels, corpus, search_kwargs=search_kwargs)
    result["ram_mb"] = {
        "baseline": round(base_rss, 1),            # interpreter + imports
        "after_load": round(after_rss, 1),         # + index + model + corpus
        "index_delta": round(after_rss - base_rss, 1),  # the index's own footprint
    }
    return result


def _print_summary(name: str, r: dict) -> None:
    print(f"  Recall@{config.TOP_K}={r[f'recall@{config.TOP_K}']:.3f}  "
          f"MRR={r['mrr']:.3f}  "
          f"warm p95={r['latency_ms']['warm_p95']:.1f}ms  "
          f"index RAM={r['ram_mb']['index_delta']:.0f}MB  "
          f"{'PASS' if r['passes_latency_50ms'] else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid", "all"],
                        default="all")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "bench.json"))
    parser.add_argument("--cooldown", type=int, default=30,
                        help="seconds to let the CPU cool between retrievers, so a "
                             "heavy retriever's heat doesn't throttle the next "
                             "one's latency. Only used with --retriever all.")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # "all" fans out to one subprocess per retriever so each is measured from a
    # clean process: isolated RAM attribution and (with the cooldown) a cool CPU
    # for latency. Each child writes its own fragment, which we merge here.
    if args.retriever == "all":
        import subprocess
        merged = {}
        for i, name in enumerate(["bm25", "dense", "hybrid"]):
            if i > 0 and args.cooldown > 0:
                print(f"  (cooling down {args.cooldown}s before {name}...)")
                time.sleep(args.cooldown)
            tmp = config.RESULTS_DIR / f"_bench_{name}.json"
            print(f"Evaluating {name} (isolated process)...")
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--retriever", name, "--output", str(tmp)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(proc.stdout)
                print(proc.stderr)
                raise SystemExit(f"{name} evaluation failed")
            merged[name] = json.load(open(tmp))[name]
            _print_summary(name, merged[name])
            tmp.unlink()
            print()
        with open(args.output, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Results written to {args.output}")
        return

    # Single-retriever path (also what each subprocess runs).
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(config.DATA_DIR)
    ).load(split="dev")
    result = _benchmark_one(args.retriever, queries, qrels, corpus)
    _print_summary(args.retriever, result)
    with open(args.output, "w") as f:
        json.dump({args.retriever: result}, f, indent=2)


if __name__ == "__main__":
    main()
