import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beir import util
from beir.datasets.data_loader import GenericDataLoader
import config
from index.bm25_indexer import BM25Indexer


def load_or_download_corpus() -> dict:
    if not config.DATA_DIR.exists():
        print("Downloading FiQA dataset...")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
        util.download_and_unzip(url, str(config.DATA_DIR.parent))
    corpus, _, _ = GenericDataLoader(data_folder=str(config.DATA_DIR)).load(split="dev")
    print(corpus)
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-only", action="store_true")
    args = parser.parse_args()

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpus...")
    corpus = load_or_download_corpus()

    print(f"Corpus size: {len(corpus)} passages")
    with open(config.CORPUS_CACHE_PATH, "wb") as f:
        pickle.dump(corpus, f)
    print(f"Corpus cache saved -> {config.CORPUS_CACHE_PATH}")

    print("\nBuilding BM25 index...")
    bm25 = BM25Indexer()
    bm25.build(corpus)
    bm25.save()
    print(f"BM25 index saved -> {config.BM25_INDEX_PATH}")

    if not args.bm25_only:
        from index.dense_indexer import DenseIndexer
        print("\nBuilding dense index...")
        dense = DenseIndexer()
        dense.build(corpus)
        dense.save()
        print(f"Dense index saved -> {config.DENSE_INDEX_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
