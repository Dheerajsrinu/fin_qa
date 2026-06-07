# Design Doc — Constrained Retrieval System over FiQA

Retrieval over BEIR/FiQA (57,638 finance Q&A passages, 500 dev queries) under three
binding constraints: **CPU-only**, **p95 ≤ 50 ms** (after warmup), **index ≤ 2 GB**.

**Hardware measured on:** 13th Gen Intel Core i5-1340P (12 physical / 16 logical
cores), 15.6 GB RAM, Windows 11, Python 3.12, CPU-only PyTorch 2.8. Quality metrics
are hardware-independent; latency is as measured on this machine.

---

## 1. Chosen operating point

**Hybrid (linear interpolation, α=0.3) = bge-small-en-v1.5 (ONNX encode) + BM25 (bm25s), fused over 50 candidates each, FAISS HNSW (M=16, efSearch=50).**

`score = 0.3 · minmax(BM25) + 0.7 · minmax(dense)`

| | Value | Constraint | Status |
|---|---|---|---|
| Recall@10 | **0.658** | — | — |
| MRR | **0.484** | — | — |
| Warm p95 | **15.4 ms** | ≤ 50 ms | ✅ |
| Index RAM (delta) | **376 MB** | ≤ 2048 MB | ✅ |
| Hardware | CPU-only | CPU-only | ✅ |

**Why this point.** It has the best Recall@10 and MRR of every configuration that
satisfies all three constraints, with ~3× latency headroom. The two enabling
decisions were (a) **bge-small** — the only retrieval-tuned model that fits the
budget; (b) **ONNX Runtime** for query encode, which halved encode latency (44 → 22 ms
p95) with *identical* embeddings (cosine 1.0 vs PyTorch), turning the linear hybrid
from a latency failure (62 ms) into the best feasible point. The hybrid's edge over
dense-alone is modest (+0.006 R@10, +0.014 MRR) but essentially free (BM25 adds ~1 ms).

**Honest caveat:** if Recall@10 were the *only* metric, dense-alone — or even MiniLM —
would be a defensible, simpler choice (see §4). We keep the hybrid for its MRR.

---

## 2. Benchmark table

All on FiQA dev (500 queries), CPU-only, ONNX dense encode. Latency measured with a
100-query warmup; each retriever benchmarked in an isolated process from a cooled CPU
(see §3). Raw output: `results/bench.json`.

| Config | Recall@10 | MRR | Cold p50/p95 | Warm p50/p95 | Index RAM |
|---|---|---|---|---|---|
| BM25 (bm25s) | 0.444 | 0.288 | 1.9 / 3.8 ms | 1.9 / 3.8 ms | 112 MB |
| Dense (bge-small) | 0.652 | 0.470 | 6.8 / 12.8 ms | 7.1 / 10.6 ms | 327 MB |
| **Hybrid (linear α=0.3)** | **0.658** | **0.484** | 10.2 / 13.4 ms | 10.1 / 15.4 ms | 376 MB |
| Ablation: Dense (MiniLM-L6) | 0.648 | 0.443 | — | 3.7 / 5.3 ms | ~325 MB |

**Stratified — Recall@10 by query length** (token = whitespace word):

| Config | short (<5, n=13) | medium (5–15, n=403) | long (>15, n=84) |
|---|---|---|---|
| BM25 | 0.154 | 0.432 | 0.548 |
| Dense | 0.462 | 0.633 | 0.774 |
| Hybrid | 0.462 | 0.640 | 0.774 |

**Stratified — Recall@10 by gold-doc length** (long = top 10% longest docs, ≥270 words):

| Config | gold in top-10% longest (n=185) | rest (n=315) |
|---|---|---|
| BM25 | 0.497 | 0.413 |
| Dense | 0.697 | 0.625 |
| Hybrid | 0.708 | 0.629 |

Two consistent patterns: **(1) recall rises monotonically with query length** — short
queries are the hard subpopulation for every retriever (fewer terms → less signal);
**(2) every retriever does better when the gold passage is a long document** — more
text gives both lexical and semantic matching more to grip. BM25 gains most from long
gold docs (+0.084), as expected for a term-frequency method.

---

## 3. Cold vs. warm latency

The cold/warm delta is **small** (hybrid cold p95 13.4 ms vs warm 15.4 ms; the
direction even reverses within noise). Reason: the expensive cold cost — loading model
weights and the FAISS index into RAM — happens at process startup, *before* the first
query, so it isn't captured in per-query timing. What remains to "warm up" is just the
ONNX graph and OS page cache, which stabilise within a few queries. This is specific to
this stack; a heavier model would show a larger delta.

**Measurement lesson (the real story).** Getting trustworthy p95 on a CPU was the
hard part. Two pitfalls, both found while building the harness:
1. Measuring latency *after* the 1500-query quality sweep captured thermal throttling —
   hybrid read **234 ms** instead of ~15 ms. Fix: time latency right after load.
2. Even then, a heavy retriever heats the chip and throttles the *next* one's
   measurement (hybrid read 133 ms straight after the dense sweep, vs 14.6 ms alone).
   Fix: `eval.py` runs each retriever in its own process with a 30 s CPU cooldown.

Takeaway: CPU latency numbers only reproduce if every measurement starts from a
comparable thermal state.

---

## 4. One counterintuitive finding

**Expected:** the retrieval-fine-tuned **bge-small** (33M) would clearly beat the
generic, general-purpose **all-MiniLM-L6-v2** (22M) on retrieval quality — that's the
whole point of retrieval-specific training.

**Observed (ablation, identical setup, ONNX both, cos=1.0):**

| | MiniLM-L6 | bge-small | Δ |
|---|---|---|---|
| Recall@10 | 0.648 | 0.652 | **+0.004** |
| MRR | 0.443 | 0.470 | **+0.027** |

On **Recall@10 they are essentially tied** — the "better" model adds 0.004. The real
gap is **MRR (+0.027)**.

**Hypothesis:** Recall@10 only asks "is the gold passage somewhere in the top 10?" —
both models are about equally good at *finding* the relevant region of the corpus. The
difference is *ordering within* that region: bge-small's retrieval fine-tuning sharpens
ranking, pushing the right passage higher (MRR), without changing coverage much. The
practical implication is real: if a downstream reader only consumes the top-1 passage,
that MRR gap matters more than the flat Recall@10 suggests — but it's a smaller,
narrower win than "retrieval-tuned beats generic" would lead you to expect. (The models
even split by query length: MiniLM wins on short queries, bge-small on long — see
`results/ablation.md`.)

---

## 5. Approaches that didn't pan out

**(a) `rank_bm25` for BM25.** *Tried:* the popular pure-Python BM25Okapi. *Expected:*
fine for 57K docs. *Observed:* warm p95 = **448 ms**, ~9× over budget. *Why:*
`get_scores()` loops over all 57K docs in Python per query — no vectorisation. *Fix:*
switched to `bm25s` (scipy sparse matrix; one C-level dot product per query) → p95
0.9 ms, identical scores, ~400× faster.

**(b) RRF hybrid fusion.** *Tried:* Reciprocal Rank Fusion, the standard "just works"
hybrid method, swept k ∈ {10,30,60,100}. *Expected:* hybrid beats either retriever
alone (the textbook result). *Observed:* RRF **hurt** quality at every k (best R@10
0.634 < 0.652 dense-alone). *Why:* FiQA is semantically driven and dense is much the
stronger retriever; RRF uses only rank position, so it lets BM25's noisier ranking pull
good dense hits *down*. Linear interpolation with a small BM25 weight (α=0.3) worked
instead because it can keep BM25 a minority voice.

**(c) mpnet-base (the highest-quality model), even via ONNX.** *Tried:* all-mpnet-base
(R@10=0.768, the best of the four candidates) with ONNX, then INT8, to beat the 50 ms
budget. *Expected:* ONNX's ~2× would bring it under 50 ms like it did for bge-small.
*Observed:* fp32 ONNX p95 = **86 ms** (median halved 77→42 ms, but the tail stayed
high — it's compute-bound at 110M params); INT8 was fast (27 ms) but quantization drift
(cos 0.76) collapsed quality to R@10 0.678 ≈ MiniLM. *Conclusion:* no mpnet config fits
50 ms while keeping its quality edge on CPU — that +0.11 R@10 is a GPU-only prize.

---

## 6. Trade-offs against the constraints

**If latency halved (p95 ≤ 25 ms):** no change — the hybrid already sits at 15 ms, well
under 25. We'd keep it. Only below ~12 ms would we have to act, at which point dropping
to dense-alone (10.6 ms) or MiniLM (5.3 ms, −0.004 R@10) buys headroom cheaply.

**If a GPU budget were available:** the picture changes most here. All four candidate
models pass latency on GPU, so we'd switch the dense model to **mpnet-base**
(R@10 0.768, MRR 0.594) — a **+0.11 R@10** jump over the CPU-bound point — and could
add a cross-encoder reranker over the top-50 (see §7 / failures), which our failure
analysis shows would fix the bulk of the *ranking* failures.

**If memory were the tight constraint:** non-binding here — even the largest config
(bge-base) used <900 MB, far under 2 GB. Latency is the binding constraint, not memory.

---

## 7. Production concerns

- **Index freshness / updates.** New FiQA-style posts need re-encoding. BM25 rebuilds
  in seconds; the dense index re-encodes 57K docs in ~10–15 min on CPU. For streaming
  updates, keep a small incremental BM25 + FAISS shard for new docs and merge
  periodically rather than full rebuilds.
- **Recall ceiling, not ranking, is the limiter.** Failure analysis (`results/failures.md`)
  shows 104/500 queries are *hard misses* (gold not even in the top-50 candidates),
  capping recall@50 at ~0.79. A reranker helps the 99 *ranking* failures but not these;
  improving first-stage recall (stronger encoder / query expansion) is the real lever.
- **Latency tail under load.** p95 was measured single-query from a cool CPU. Under
  sustained concurrent load this laptop-class chip throttles (we saw 2–15× slowdowns).
  Production sizing must budget for thermal/contention headroom and pin thread counts.
- **Monitoring.** Track daily p95 latency (detects model/index/host regressions),
  Recall@10 on a labeled canary query set (detects quality drift after re-index), and
  RSS (detects leaks as the corpus grows). Alert on the short-query and hard-miss rates
  specifically — they're the known weak spots.
- **Cost / scaling.** HNSW RAM grows ~linearly (≈880 MB at 10× corpus, still <2 GB) and
  search is O(log N), so the index scales well; the encode step dominates cost and is
  the first thing to move to a GPU or a batched/quantized server if QPS grows.

---

*Ablations: `results/ablation.md`. Failure cases: `results/failures.md`.*
