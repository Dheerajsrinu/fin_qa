"""
ABLATION: dense model choice — all-MiniLM-L6-v2 vs bge-small-en-v1.5.

One variable changed on top of the chosen operating point: swap the dense encoder
(and its FAISS index), hold everything else fixed (ONNX backend, HNSW efSearch=50,
top_k=10, same 500 FiQA dev queries). Both are small CPU-friendly models, so the
constraint picture doesn't move — this isolates pure quality, which answers:
does bge-small's retrieval-specific fine-tuning actually beat the hugely popular
general-purpose MiniLM on this financial corpus?

MiniLM uses MEAN pooling, bge-small uses CLS — handled by the encoder. Both run on
ONNX so the latency comparison is apples-to-apples. We also verify each model's
ONNX embeddings match its PyTorch ones (the FAISS indexes were built in PyTorch).

Run:
    PYTHONPATH=src python experiments/ablation_dense_model.py
"""

import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

import config
from eval import recall_at_k, mrr            # reuse the exact graded metrics
from retrieval.onnx_encoder import OnnxEncoder


MODELS = [
    {"label": "MiniLM-L6 (baseline)", "hf_id": "sentence-transformers/all-MiniLM-L6-v2",
     "onnx": config.INDEX_DIR / "minilm_onnx" / "model.onnx",
     "faiss": config.INDEX_DIR / "minilm.faiss", "pooling": "mean", "max_len": 256,
     "params": "22M"},
    {"label": "bge-small (chosen)", "hf_id": "BAAI/bge-small-en-v1.5",
     "onnx": config.INDEX_DIR / "bge_small_onnx" / "model.onnx",
     "faiss": config.INDEX_DIR / "bge_small.faiss", "pooling": "cls", "max_len": 512,
     "params": "33M"},
]
COOLDOWN_S = 20      # let the CPU cool between models so latency is comparable


def quality(search_fn, qids, qts, qrels, items=None):
    pairs = items if items is not None else list(zip(qids, qts))
    rc, mr = [], []
    for qid, qt in pairs:
        rel = set(qrels.get(qid, {}).keys())
        got = search_fn(qt)
        rc.append(recall_at_k(got, rel, k=10))
        mr.append(mrr(got, rel))
    return float(np.mean(rc)), float(np.mean(mr)), len(pairs)


def warm_latency(search_fn, qts, warm=100, n=100):
    for q in qts[:warm]:
        search_fn(q)
    s = []
    for q in qts[warm:warm + n]:
        t = time.perf_counter()
        search_fn(q)
        s.append((time.perf_counter() - t) * 1000)
    return float(np.percentile(s, 50)), float(np.percentile(s, 95))


print("Loading FiQA dev set...")
_, queries, qrels = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")
qids, qts = list(queries.keys()), list(queries.values())
doc_ids = pickle.load(open(config.DOC_IDS_PATH, "rb"))

# stratify by median query length (same split eval.py uses)
lengths = [len(t.split()) for t in qts]
thr = float(np.median(lengths))
short = [(qid, qt) for qid, qt, n in zip(qids, qts, lengths) if n <= thr]
long_ = [(qid, qt) for qid, qt, n in zip(qids, qts, lengths) if n > thr]
print(f"  {len(qts)} queries (short<= {thr:.0f}w: {len(short)}, long: {len(long_)})\n")

results = []
for i, m in enumerate(MODELS):
    if i > 0:
        print(f"(cooling down {COOLDOWN_S}s...)\n")
        time.sleep(COOLDOWN_S)

    print("=" * 60)
    print(f"{m['label']}  ({m['params']} params, pooling={m['pooling']})")
    print("=" * 60)

    enc = OnnxEncoder(m["onnx"], pooling=m["pooling"], max_seq_len=m["max_len"])
    index = faiss.read_index(str(m["faiss"]))
    index.hnsw.efSearch = config.HNSW_EF_SEARCH

    # ONNX-vs-PyTorch agreement (recall is only valid if these match the index)
    pt = SentenceTransformer(m["hf_id"])
    sample = qts[:200]
    cos = np.sum(pt.encode(sample, normalize_embeddings=True, convert_to_numpy=True)
                 * np.vstack([enc.encode(q) for q in sample]), axis=1)
    print(f"agreement vs PyTorch: mean cos={cos.mean():.5f}")
    del pt

    def search(q, top_k=10):
        v = enc.encode(q, normalize_embeddings=True)
        _, idx = index.search(v, top_k)
        return [doc_ids[j] for j in idx[0] if j >= 0]

    # latency FIRST (cool CPU), then the heavy quality sweep
    p50, p95 = warm_latency(search, qts)

    r10, mrr_all, _ = quality(search, qids, qts, qrels)
    sr10, smrr, sn  = quality(search, None, None, qrels, items=short)
    lr10, lmrr, ln  = quality(search, None, None, qrels, items=long_)

    print(f"R@10={r10:.3f}  MRR={mrr_all:.3f}  warm p50={p50:.1f}ms p95={p95:.1f}ms")
    print(f"  short R@10={sr10:.3f} MRR={smrr:.3f}  |  long R@10={lr10:.3f} MRR={lmrr:.3f}\n")

    results.append({"label": m["label"], "params": m["params"], "r10": r10,
                    "mrr": mrr_all, "p50": p50, "p95": p95,
                    "sr10": sr10, "lr10": lr10})
    del index, enc
    import gc; gc.collect()


# ── deltas ────────────────────────────────────────────────────────────────────
base, chosen = results[0], results[1]      # MiniLM is the baseline being ablated
print("=" * 70)
print("ABLATION RESULT — dense model: MiniLM-L6 -> bge-small")
print("=" * 70)
print(f"{'Metric':<18}{'MiniLM-L6':>12}{'bge-small':>12}{'delta':>12}")
print("-" * 70)
rows = [
    ("Recall@10",      base["r10"],  chosen["r10"]),
    ("MRR",            base["mrr"],  chosen["mrr"]),
    ("R@10 (short q)", base["sr10"], chosen["sr10"]),
    ("R@10 (long q)",  base["lr10"], chosen["lr10"]),
    ("Warm p50 (ms)",  base["p50"],  chosen["p50"]),
    ("Warm p95 (ms)",  base["p95"],  chosen["p95"]),
]
for name, b, c in rows:
    print(f"{name:<18}{b:>12.3f}{c:>12.3f}{c - b:>+12.3f}")
print("=" * 70)
