"""
Compare query-encode backends for bge-small: PyTorch vs ONNX (fp32) vs ONNX (int8).

The dense pipeline's latency is dominated by the query encode step (~37ms warm in
PyTorch); FAISS search is ~1ms. So encode latency is what determines whether the
linear hybrid can fit under the 50ms p95 budget. This script measures, on real
FiQA queries:

  1. Embedding AGREEMENT vs PyTorch (cosine similarity). The FAISS index was built
     with PyTorch embeddings, so a backend whose query vectors drift from PyTorch
     will silently lose recall. We need agreement ~0.9999 for fp32; int8 will be
     a touch lower and we judge whether it's acceptable.
  2. Single-query encode LATENCY: warm p50/p95 after a 100-query warmup.

Run:
    PYTHONPATH=src python experiments/compare_onnx_dense.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer

import config
from retrieval.onnx_encoder import OnnxEncoder, ONNX_DIR


def warm_latency(encode_fn, queries, warmup=100, measure=100):
    for q in queries[:warmup]:
        encode_fn(q)
    samples = []
    for q in queries[warmup:warmup + measure]:
        t0 = time.perf_counter()
        encode_fn(q)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 95))


def agreement(emb_a, emb_b):
    """Mean per-row cosine similarity between two (n, d) sets of unit vectors."""
    cos = np.sum(emb_a * emb_b, axis=1)
    return float(cos.mean()), float(cos.min())


print("Loading FiQA dev queries...")
_, queries, _ = GenericDataLoader(str(config.DATA_DIR)).load(split="dev")
query_texts = list(queries.values())
print(f"  {len(query_texts)} queries\n")

# Reference embeddings from PyTorch on a sample, for agreement checks.
sample = query_texts[:200]

print("Loading backends...")
pt = SentenceTransformer(config.DENSE_MODEL)
onnx_fp32 = OnnxEncoder(ONNX_DIR / "model.onnx")
onnx_int8 = OnnxEncoder(ONNX_DIR / "model_int8.onnx")

pt_encode   = lambda q: pt.encode([q], normalize_embeddings=True, convert_to_numpy=True)
fp32_encode = lambda q: onnx_fp32.encode(q, normalize_embeddings=True)
int8_encode = lambda q: onnx_int8.encode(q, normalize_embeddings=True)

# ── Embedding agreement vs PyTorch ────────────────────────────────────────────
print("\n" + "=" * 60)
print("EMBEDDING AGREEMENT vs PyTorch (cosine, higher = safer recall)")
print("=" * 60)
pt_emb   = pt.encode(sample, normalize_embeddings=True, convert_to_numpy=True)
fp32_emb = np.vstack([onnx_fp32.encode(q) for q in sample])
int8_emb = np.vstack([onnx_int8.encode(q) for q in sample])

for name, emb in [("ONNX fp32", fp32_emb), ("ONNX int8", int8_emb)]:
    mean_cos, min_cos = agreement(pt_emb, emb)
    print(f"  {name:<12}  mean cos={mean_cos:.5f}  min cos={min_cos:.5f}")

# ── Encode latency ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ENCODE LATENCY (single query, warm: 100-query warmup + 100 measured)")
print("=" * 60)
print(f"{'Backend':<14}  {'p50':>8}  {'p95':>8}")
print("-" * 36)
for name, fn in [("PyTorch", pt_encode), ("ONNX fp32", fp32_encode), ("ONNX int8", int8_encode)]:
    p50, p95 = warm_latency(fn, query_texts)
    print(f"{name:<14}  {p50:>6.1f}ms  {p95:>6.1f}ms")

print("\nNote: full dense search ~= encode + ~1ms FAISS. Encode is the bottleneck.")
