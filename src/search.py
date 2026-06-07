import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
import config


def build_retriever(method: str):
    """Return (retriever, search_kwargs). Hybrid defaults to the chosen operating
    point: linear fusion at alpha=0.3 (see DESIGN / results/ablation.md)."""
    if method == "bm25":
        return BM25Retriever.load(), {}
    if method == "dense":
        return DenseRetriever.load(), {}
    if method == "hybrid":
        return HybridRetriever.load(), {"method": "linear", "alpha": 0.3}
    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid"], default="hybrid")
    args = parser.parse_args()

    retriever, search_kwargs = build_retriever(args.method)
    results = retriever.search(args.query, top_k=args.top_k, **search_kwargs)

    for rank, (doc_id, score, text) in enumerate(results, 1):
        print(f"[{rank}] score={score:.4f}  id={doc_id}")
        print(f"    {text[:200]}")
        print()


if __name__ == "__main__":
    main()
