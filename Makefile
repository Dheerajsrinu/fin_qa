PYTHON     := python
SRC        := src
export PYTHONPATH := $(SRC)

.PHONY: all index eval bench test ablation failures clean search

all: index eval

# Downloads FiQA, builds BM25 + dense (FAISS HNSW) indexes, and exports the
# serve-time ONNX encoder. Dense encode of 57K passages on CPU takes ~10-15 min.
index:
	$(PYTHON) $(SRC)/index/build_index.py

# Benchmarks BM25, dense, hybrid on the FiQA dev set -> results/bench.json
# (each retriever runs in its own process with a CPU cooldown for clean latency).
eval:
	$(PYTHON) $(SRC)/eval.py --output results/bench.json

bench: eval

test:
	pytest tests/ -v

# Step 6 ablation (MiniLM vs bge-small). Requires the MiniLM ONNX export.
ablation:
	$(PYTHON) $(SRC)/index/export_onnx.py --model minilm
	$(PYTHON) experiments/ablation_dense_model.py

# Step 7 failure analysis dump.
failures:
	$(PYTHON) experiments/find_failures.py

# QUERY="..." [K=10] [METHOD=hybrid]
search:
	$(PYTHON) $(SRC)/search.py --query "$(QUERY)" --top-k $(or $(K),10) --method $(or $(METHOD),hybrid)

clean:
	rm -f indexes/*.pkl indexes/*.faiss indexes/*.npy
	rm -rf indexes/bge_small_onnx indexes/mpnet_onnx indexes/minilm_onnx
