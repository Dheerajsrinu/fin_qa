"""
Find and dump failure cases for the chosen system (hybrid linear, alpha=0.3).

A "failure" = a gold passage that does not appear in the top 5. We bucket failures
into (a) hard misses (gold not even in top 50 candidates) and (b) ranking failures
(gold retrieved but buried below rank 5), and dump the query, the gold passage, and
the top-5 retrieved passages so we can diagnose each by eye.

Run:
    PYTHONPATH=src python experiments/find_failures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
import config
from retrieval.hybrid import HybridRetriever


def gold_rank(retrieved_ids, gold_ids):
    for r, d in enumerate(retrieved_ids, 1):
        if d in gold_ids:
            return r
    return None


print("Loading corpus + retriever (hybrid linear a=0.3)...")
corpus, queries, qrels = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")
h = HybridRetriever.load()

failures = []
for qid, qtext in queries.items():
    gold_ids = set(qrels.get(qid, {}).keys())
    if not gold_ids:
        continue
    # fetch deep (50) so we can tell a hard miss from a ranking failure
    results = h.search(qtext, top_k=50, method="linear", alpha=0.3)
    ids = [d for d, _, _ in results]
    rank = gold_rank(ids, gold_ids)
    if rank is None or rank > 5:                 # not in top 5 => failure
        failures.append({
            "qid": qid, "qtext": qtext, "gold_ids": gold_ids,
            "rank": rank, "results": results,
            "n_words": len(qtext.split()),
        })

hard = [f for f in failures if f["rank"] is None]
ranking = [f for f in failures if f["rank"] is not None]
print(f"\nTotal failures (gold not in top 5): {len(failures)} / {len(queries)}")
print(f"  hard misses (gold not in top 50): {len(hard)}")
print(f"  ranking failures (gold in 6..50): {len(ranking)}")
print(f"  short-query failures (<=10 words): {sum(1 for f in failures if f['n_words'] <= 10)}")

# Sort ranking failures by how deep the gold sits (worst first) for variety.
ranking.sort(key=lambda f: f["rank"], reverse=True)


def dump(f, tag):
    print("\n" + "#" * 78)
    print(f"{tag}  qid={f['qid']}  |  query ({f['n_words']} words): {f['qtext']!r}")
    print(f"gold rank: {f['rank'] if f['rank'] else 'NOT in top 50 (hard miss)'}")
    for gid in f["gold_ids"]:
        gdoc = corpus.get(gid, {})
        gtext = (gdoc.get("title", "") + " " + gdoc.get("text", "")).strip()
        print(f"\n  GOLD [{gid}]: {gtext[:400]}")
    print("\n  --- top 5 retrieved ---")
    for i, (d, score, text) in enumerate(f["results"][:5], 1):
        mark = "  <== GOLD" if d in f["gold_ids"] else ""
        print(f"  {i}. [{d}] score={score:.4f}{mark}\n     {text[:240]}")


# Dump a spread: a few hard misses and a few deep ranking failures.
for f in hard[:3]:
    dump(f, "HARD MISS")
for f in ranking[:4]:
    dump(f, "RANKING FAILURE")
