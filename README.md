# Constrained Retrieval System — FiQA

BM25 + Dense + Hybrid retrieval over the BEIR/FiQA financial Q&A corpus (57,638
passages, 500 dev queries). **CPU-only, p95 ≤ 50 ms after warmup, index ≤ 2 GB RAM.**

**Chosen operating point:** hybrid linear fusion (α=0.3) = bge-small-en-v1.5 (encoded
via **ONNX Runtime** for ~2× faster CPU inference) + BM25 (bm25s), fused over FAISS
HNSW candidates. Recall@10 = 0.658, MRR = 0.484, warm p95 ≈ 15 ms, index RAM ≈ 376 MB.
See `DESIGN.md` for the full justification.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Build indexes (run once)

Downloads FiQA, builds the BM25 + dense (FAISS HNSW) indexes, and exports the
serve-time ONNX encoder. Dense encode of 57K passages on CPU takes ~10–15 min.

```bash
make index
# or: PYTHONPATH=src python src/index/build_index.py
```

> A pre-built index artifact is shipped under `indexes/` so you can skip this and go
> straight to search/eval; `make index` regenerates it from scratch. Because the ONNX
> embeddings are identical to PyTorch (cosine 1.0), a rebuild reproduces the same
> quality numbers.

### Search (CLI)

```bash
python src/search.py --query "what is short selling?" --top-k 10 --method hybrid
# method: bm25 | dense | hybrid   (hybrid = linear fusion, alpha=0.3)
# or: make search QUERY="what is short selling?" K=10 METHOD=hybrid
```

### Evaluate

```bash
make eval        # writes results/bench.json (BM25, dense, hybrid + stratified metrics)
```

Each retriever is benchmarked in its own process with a CPU cooldown between them, so
latency isn't contaminated by thermal throttling (see DESIGN §3). The full run takes a
few minutes; pass `--cooldown 0` to `src/eval.py` for a quick (less accurate) pass.

### Ablation & failure analysis

```bash
make ablation    # MiniLM-L6 vs bge-small dense-model ablation -> see results/ablation.md
make failures    # dumps failure cases -> see results/failures.md
```

### Tests

```bash
make test        # unit tests for rrf_score, recall_at_k, mrr, hybrid scoring, etc.
```

## Docker (reproducible runtime)

```bash
# Step 1 — Build the image
docker build -t rag-fiqa .

# Build indexes inside the container (writes to mounted ./indexes)
docker run --rm -v $(pwd)/indexes:/app/indexes rag-fiqa make index

# Run the evaluation
docker run --rm -v ${PWD}/results:/app/results rag-fiqa make eval

# Try a search query in the container
docker run --rm rag-fiqa make search QUERY="what is short selling?" K=5 METHOD=hybrid

# Run eval (reads indexes, writes results)
docker run --rm \
  -v $(pwd)/indexes:/app/indexes \
  -v $(pwd)/results:/app/results \
  rag-fiqa make eval
```
