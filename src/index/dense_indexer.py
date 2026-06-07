import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


class DenseIndexer:
    def __init__(self, model_name: str = config.DENSE_MODEL):
        print(f"Loading model {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        self.doc_ids: list[str] = []

    def build(self, corpus: dict) -> None:
        self.doc_ids = list(corpus.keys())
        texts = [doc["title"] + " " + doc["text"] for doc in corpus.values()]

        print(f"Encoding {len(texts)} passages on CPU (takes ~10-15 min) ...")
        embeddings = self.model.encode(
            texts,
            batch_size=config.DENSE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,   # unit vectors so inner product == cosine
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]        # 384 for bge-small-en-v1.5
        self.index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        self.index.add(embeddings)
        print(f"HNSW index: {self.index.ntotal} vectors, dim={dim}")

    def save(self, path: Path = config.DENSE_INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        with open(config.DOC_IDS_PATH, "wb") as f:
            pickle.dump(self.doc_ids, f)
