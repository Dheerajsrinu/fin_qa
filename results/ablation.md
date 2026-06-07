# Ablations

Two real ablations were run on top of the three baselines (BM25, dense, hybrid).
Each changes **one variable**, holds everything else fixed, and reports the
observed deltas with an explanation.

1. **Dense model choice** — MiniLM-L6 vs bge-small (which encoder?)
2. **Hybrid weighting** — the linear-interpolation α (how much BM25 vs dense?)

All numbers are on the 500 FiQA dev queries, ONNX backend, HNSW `efSearch=50`,
`CANDIDATE_K=50`, `top_k=10`.

---

## Ablation 1 — Dense model: MiniLM-L6 vs bge-small

Swap the dense encoder and its FAISS index, keep everything else fixed. Both are
small CPU-friendly models, so the latency/RAM constraints hold either way — this
isolates **pure retrieval quality**. Question: does bge-small's retrieval-specific
fine-tuning actually beat the popular general-purpose all-MiniLM-L6-v2 here?

Both models' ONNX embeddings match their PyTorch ones (mean cos = 1.00000), so the
indexes are valid and the deltas are real model differences, not artifacts.

| Metric | MiniLM-L6 (22M) | bge-small (33M) | Δ (bge − MiniLM) |
|---|---|---|---|
| **Recall@10** | 0.648 | 0.652 | **+0.004** |
| **MRR** | 0.443 | 0.470 | **+0.027** |
| Recall@10 — short queries (≤10w) | 0.600 | 0.585 | −0.015 |
| Recall@10 — long queries (>10w) | 0.700 | 0.725 | +0.025 |
| Warm p50 latency | 3.7ms | 6.9ms | +3.2ms |
| Warm p95 latency | 5.3ms | 9.7ms | +4.4ms |

**Explanation.**

1. **Almost no Recall@10 difference (+0.004), but a real MRR gain (+0.027).** The
   "better", larger, retrieval-tuned model is essentially tied with generic MiniLM
   on whether the gold passage lands in the top 10. Where it pulls ahead is
   *ranking* — MRR measures how high the first relevant result sits, and bge-small
   reliably ranks it nearer the top. Both models find the relevant region equally
   well (coverage); bge-small's fine-tuning sharpens the ordering within it. If a
   RAG reader only consumes the top passage, that +0.027 MRR matters more than the
   flat Recall@10 suggests.

2. **The models split by query length.** MiniLM is *better* on short queries
   (0.600 vs 0.585); bge-small wins on long ones (0.725 vs 0.700). Terse,
   keyword-like FiQA queries ("401k rollover") give bge-small's retrieval tuning
   little to exploit; the extra context in long queries does. This reinforces the
   broader finding that short queries are the hard subpopulation.

3. **Cost: bge-small is ~2× slower (5.3 → 9.7ms p95), still trivially under 50ms.**
   MiniLM is faster because it is smaller (22M vs 33M) and truncates at 256 tokens
   vs 512. At these absolute latencies it doesn't drive the decision.

**Decision.** Keep **bge-small**. Its edge is modest on Recall@10 but consistent on
MRR (+0.027) and long-query recall (+0.025), for ~4ms p95. Honest caveat: if
Recall@10 were the only graded metric, **MiniLM is a defensible ~2× faster choice**
at near-identical quality. bge-small earns its place through ranking quality.

---

## Ablation 2 — Hybrid weighting: linear-interpolation α

`score = α · norm(BM25) + (1−α) · norm(dense)`. Sweep α from pure dense (0.0) to
pure BM25 (1.0), holding the dense model (bge-small) and BM25 index fixed. This is
the knob that decides how much lexical vs semantic signal the hybrid trusts.

| α | R@10 | MRR | Note |
|---|---|---|---|
| 0.0 | 0.652 | 0.470 | pure dense (baseline) |
| 0.1 | 0.654 | 0.475 | |
| 0.2 | 0.654 | 0.482 | |
| **0.3** | **0.658** | **0.484** | **best — chosen operating point** |
| 0.4 | 0.652 | 0.477 | |
| 0.5 | 0.642 | 0.447 | |
| 0.6 | 0.624 | 0.398 | |
| 0.7 | 0.550 | 0.361 | |
| 0.8 | 0.508 | 0.333 | |
| 0.9 | 0.478 | 0.306 | |
| 1.0 | 0.444 | 0.288 | pure BM25 |

**Deltas vs pure dense (α=0.0):** best α=0.3 gives **+0.006 Recall@10** and
**+0.014 MRR**. The peak is shallow — any α in [0.1, 0.4] is within ~0.006 of the
best — then quality falls off a cliff past α=0.5.

**Explanation.**

1. **The optimum is a *minority* BM25 weight (~30%).** FiQA is a
   semantically-driven corpus (paraphrased finance Q&A), so dense should dominate.
   BM25 contributes a small complementary signal — exact tokens like tickers,
   "401k", "ETF" — that dense sometimes under-weights. As a 30% partner it nudges a
   few of those keyword-exact passages up; as a majority partner it drags in its
   weaker keyword-only ranking.

2. **Asymmetric, non-linear falloff.** Going from α=0.3 → 0.0 costs almost nothing
   (0.658 → 0.652), but α=0.3 → 0.5 already starts degrading and α>0.5 collapses
   toward pure-BM25 quality (0.444). The fusion is forgiving of *too little* BM25
   and punishing of *too much* — consistent with dense being the stronger base
   retriever that BM25 should only lightly correct.

3. **Small absolute gain.** The hybrid's win over pure dense is real but modest
   (+0.006 R@10 / +0.014 MRR). It's worth taking because it costs essentially
   nothing at α=0.3 (BM25 adds ~1ms), but it is not a dramatic improvement — on
   this corpus, dense alone is already most of the way there.

> We also tried **RRF** fusion (rank-based, swept k ∈ {10,30,60,100}); it *hurt*
> quality at every k (best R@10=0.634 < 0.652 dense alone), so linear interpolation
> was kept.

**Decision.** Linear fusion at **α=0.3** is the chosen hybrid operating point.

---

## Reproduce

```
# Ablation 1 (one-time MiniLM ONNX export, then run)
PYTHONPATH=src python src/index/export_onnx.py --model minilm
PYTHONPATH=src python experiments/ablation_dense_model.py

# Ablation 2 (alpha + RRF sweeps with latency)
PYTHONPATH=src python experiments/compare_hybrid_methods.py
```
