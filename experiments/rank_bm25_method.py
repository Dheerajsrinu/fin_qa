import sys
import time
from pathlib import Path

import numpy as np
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
import config


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

corpus, queries, qrels = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")

print("Tokenizing 57K passages with rank_bm25")
t0 = time.perf_counter()
doc_ids = list(corpus.keys())
tokenized_corpus = [
    tokenize(doc["title"] + " " + doc["text"])
    for doc in corpus.values()
]
print(tokenized_corpus[:3])
bm25 = BM25Okapi(tokenized_corpus)
build_time = time.perf_counter() - t0
print(f"Index built in {build_time:.1f}s")


def search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
    tokens = tokenize(query)
    # print("\n Query -> ",query,"\n tokens -> ",tokens)
    scores = bm25.get_scores(tokens)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(doc_ids[i], float(scores[i])) for i in top_indices]


query_list = list(queries.values())

print("\nMeasuring cold-start latency (first 20 queries)")
cold_times = []
for q in query_list[:20]:
    t0 = time.perf_counter()
    search(q, top_k=10)
    cold_times.append((time.perf_counter() - t0) * 1000)

print("Warming up (100 queries)...")
for q in query_list[20:120]:
    search(q, top_k=10)

print("Measuring warm latency (next 100 queries)")
warm_times = []
for q in query_list[120:220]:
    t0 = time.perf_counter()
    search(q, top_k=10)
    warm_times.append((time.perf_counter() - t0) * 1000)


# ── results ─────────────────────────────────────────────────────────────────

print("\n" + "=" * 50)
print("rank_bm25 latency results")
print("=" * 50)
print(f"Cold (first 20):  p50={np.percentile(cold_times, 50):.1f}ms  p95={np.percentile(cold_times, 95):.1f}ms")
print(f"Warm (after 100): p50={np.percentile(warm_times, 50):.1f}ms  p95={np.percentile(warm_times, 95):.1f}ms")
print(f"\nConstraint: p95 <= 50ms")
print(f"rank_bm25 passes: {'YES' if np.percentile(warm_times, 95) <= 50 else 'NO'}")

print("\n--- sample results for 'what is short selling?' ---")
for rank, (doc_id, score) in enumerate(search("what is short selling?", top_k=5), 1):
    text = corpus[doc_id]["text"]
    print(f"[{rank}] score={score:.3f}  {text[:120]}")
