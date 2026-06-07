import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
import config
from index.bm25_indexer import BM25Indexer, tokenize
from retrieval.bm25 import BM25Retriever

print("Loading corpus...")
corpus, queries, qrels = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")

print("Building bm25s index...")
t0 = time.perf_counter()
indexer = BM25Indexer()
indexer.build(corpus)
build_time = time.perf_counter() - t0
print(f"Index built in {build_time:.1f}s")

retriever = BM25Retriever(indexer, corpus)

query_list = list(queries.values())

print("\nMeasuring cold-start latency (first 20 queries)...")
cold_times = []
for q in query_list[:20]:
    t0 = time.perf_counter()
    retriever.search(q, top_k=10)
    cold_times.append((time.perf_counter() - t0) * 1000)

print("Warming up (100 queries)...")
for q in query_list[20:120]:
    retriever.search(q, top_k=10)

print("Measuring warm latency (next 100 queries)...")
warm_times = []
for q in query_list[120:220]:
    t0 = time.perf_counter()
    retriever.search(q, top_k=10)
    warm_times.append((time.perf_counter() - t0) * 1000)

print("\nComputing Recall@10 on first 50 queries...")
hits = 0
for qid, query_text in list(queries.items())[:50]:
    results = retriever.search(query_text, top_k=10)
    retrieved_ids = {doc_id for doc_id, _, _ in results}
    relevant_ids = set(qrels.get(qid, {}).keys())
    if retrieved_ids & relevant_ids:
        hits += 1
recall_50 = hits / 50

print("\n" + "=" * 50)
print("bm25s latency results")
print("=" * 50)
print(f"Cold (first 20):  p50={np.percentile(cold_times, 50):.1f}ms  p95={np.percentile(cold_times, 95):.1f}ms")
print(f"Warm (after 100): p50={np.percentile(warm_times, 50):.1f}ms  p95={np.percentile(warm_times, 95):.1f}ms")
print(f"\nRecall@10 (first 50 queries): {recall_50:.3f}")

print("\n" + "=" * 50)
print("Sample search results")
print("=" * 50)

sample_queries = [
    "what is short selling?",
    "how does inflation affect bond yields?",
    "capital gains tax on stock sales",
    "what is a Roth IRA?",
]

for q in sample_queries:
    print(f"\nQuery: {q}")
    results = retriever.search(q, top_k=5)
    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"  [{rank}] score={score:.3f}  id={doc_id}  {text[:120]}")

print("\n" + "=" * 50)
print("Tokenizer behaviour")
print("=" * 50)

examples = [
    "What is short selling?",
    "401k retirement savings plan",
    "S&P 500 index fund vs ETF",
]
for text in examples:
    print(f"  input : {text}")
    print(f"  tokens: {tokenize(text)}")
    print()
