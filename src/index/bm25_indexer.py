import pickle
import re
from pathlib import Path

import bm25s
from tqdm import tqdm

import config


def tokenize(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


class BM25Indexer:
    def __init__(self):
        self.bm25: bm25s.BM25 | None = None
        self.doc_ids: list[str] = []

    def build(self, corpus: dict) -> None:
        self.doc_ids = list(corpus.keys())
        texts = [
            doc["title"] + " " + doc["text"]
            for doc in tqdm(corpus.values(), desc="Preparing texts")
        ]
        corpus_tokens = bm25s.tokenize(texts, show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=False)

    def save(self, path: Path = config.BM25_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
