import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
import config

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def mrr_score(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def load_retriever(model_name: str, faiss_path: str):
    index = faiss.read_index(faiss_path)
    index.hnsw.efSearch = config.HNSW_EF_SEARCH
    with open(config.DOC_IDS_PATH, "rb") as f:
        doc_ids = pickle.load(f)
    with open(config.CORPUS_CACHE_PATH, "rb") as f:
        corpus = pickle.load(f)
    model = SentenceTransformer(model_name)
    return index, doc_ids, corpus, model


def search(query: str, index, doc_ids, model, top_k: int = 10):
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    scores, indices = index.search(vec, top_k)
    return [doc_ids[i] for i in indices[0] if i >= 0]

MODELS = [
    {
        "name":       "all-MiniLM-L6-v2",
        "label":      "MiniLM-L6",
        "faiss_path": "indexes/minilm.faiss",
        "params":     "22M",
        "dim":        384,
    },
    {
        "name":       "all-mpnet-base-v2",
        "label":      "mpnet-base",
        "faiss_path": "indexes/mpnet.faiss",
        "params":     "110M",
        "dim":        768,
    },
    {
        "name":       "BAAI/bge-small-en-v1.5",
        "label":      "bge-small",
        "faiss_path": "indexes/bge_small.faiss",
        "params":     "33M",
        "dim":        384,
    },
    {
        "name":       "BAAI/bge-base-en-v1.5",
        "label":      "bge-base",
        "faiss_path": "indexes/bge_base.faiss",
        "params":     "109M",
        "dim":        768,
    },
]

print("Loading FiQA dev set...")
corpus, queries, qrels = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")
query_ids   = list(queries.keys())
query_texts = list(queries.values())
print(f"Queries: {len(queries)}   Corpus: {len(corpus)}")

results = []

for m in MODELS:
    print(f"\n{'='*55}")
    print(f"Model : {m['label']}  ({m['params']} params, dim={m['dim']})")
    print(f"{'='*55}")

    ram_before = psutil.Process().memory_info().rss / 1024 / 1024
    t_load = time.perf_counter()
    index, doc_ids, corpus_cache, model = load_retriever(m["name"], m["faiss_path"])
    load_sec = time.perf_counter() - t_load
    ram_after = psutil.Process().memory_info().rss / 1024 / 1024
    ram_delta = ram_after - ram_before
    print(f"Loaded in {load_sec:.1f}s  |  RAM delta: +{ram_delta:.0f} MB  (total: {ram_after:.0f} MB)")

    cold_times = []
    for q in query_texts[:20]:
        t0 = time.perf_counter()
        search(q, index, doc_ids, model)
        cold_times.append((time.perf_counter() - t0) * 1000)
    print(f"Cold   p50={np.percentile(cold_times,50):.1f}ms  p95={np.percentile(cold_times,95):.1f}ms")

    for q in query_texts[20:120]:
        search(q, index, doc_ids, model)

    warm_times = []
    for q in query_texts[120:220]:
        t0 = time.perf_counter()
        search(q, index, doc_ids, model)
        warm_times.append((time.perf_counter() - t0) * 1000)
    warm_p50 = np.percentile(warm_times, 50)
    warm_p95 = np.percentile(warm_times, 95)
    print(f"Warm   p50={warm_p50:.1f}ms  p95={warm_p95:.1f}ms")

    recalls, mrrs = [], []
    for qid, qt in zip(query_ids, query_texts):
        retrieved = search(qt, index, doc_ids, model, top_k=10)
        relevant  = set(qrels.get(qid, {}).keys())
        recalls.append(recall_at_k(retrieved, relevant, k=10))
        mrrs.append(mrr_score(retrieved, relevant))

    r10  = np.mean(recalls)
    mrr  = np.mean(mrrs)
    print(f"Recall@10 = {r10:.3f}   MRR = {mrr:.3f}")

    passes = warm_p95 <= 50
    print(f"Latency constraint (p95<=50ms): {'PASS' if passes else 'FAIL'}")

    results.append({
        "label":    m["label"],
        "params":   m["params"],
        "dim":      m["dim"],
        "cold_p50": np.percentile(cold_times, 50),
        "cold_p95": np.percentile(cold_times, 95),
        "warm_p50": warm_p50,
        "warm_p95": warm_p95,
        "recall10": r10,
        "mrr":      mrr,
        "ram_mb":   ram_after,
        "passes":   passes,
    })

    del model, index
    import gc; gc.collect()


print("\n\n" + "="*90)
print("COMPARISON TABLE — all 4 dense models on FiQA dev (500 queries)")
print("="*90)

header = f"{'Model':<14} {'Params':>6} {'Dim':>5} {'R@10':>6} {'MRR':>6} {'Cold p50':>9} {'Cold p95':>9} {'Warm p50':>9} {'Warm p95':>9} {'RAM':>7} {'50ms?':>6}"
print(header)
print("-" * 90)

for r in results:
    flag = "YES" if r["passes"] else "NO"
    row = (
        f"{r['label']:<14} {r['params']:>6} {r['dim']:>5} "
        f"{r['recall10']:>6.3f} {r['mrr']:>6.3f} "
        f"{r['cold_p50']:>8.1f}ms {r['cold_p95']:>8.1f}ms "
        f"{r['warm_p50']:>8.1f}ms {r['warm_p95']:>8.1f}ms "
        f"{r['ram_mb']:>6.0f}MB {flag:>6}"
    )
    print(row)

print("="*90)
print()
print("Key tradeoff:")
best_quality = max(results, key=lambda x: x["recall10"])
best_speed   = min((r for r in results if r["passes"]), key=lambda x: x["warm_p95"], default=None)
print(f"  Best Recall@10 : {best_quality['label']} ({best_quality['recall10']:.3f})")
if best_speed:
    print(f"  Fastest passing: {best_speed['label']} (warm p95={best_speed['warm_p95']:.1f}ms)")
passing = [r for r in results if r["passes"]]
if passing:
    best_passing_quality = max(passing, key=lambda x: x["recall10"])
    print(f"  Best quality within constraint: {best_passing_quality['label']} (R@10={best_passing_quality['recall10']:.3f}, p95={best_passing_quality['warm_p95']:.1f}ms)")
