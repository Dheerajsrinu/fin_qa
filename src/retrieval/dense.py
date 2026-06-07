import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config


class DenseRetriever:
    def __init__(
        self,
        index: faiss.Index,
        doc_ids: list[str],
        corpus: dict,
        model: SentenceTransformer,
    ):
        self.index = index
        self.doc_ids = doc_ids
        self.corpus = corpus
        self.model = model

    @classmethod
    def load(
        cls,
        index_path: Path = config.DENSE_INDEX_PATH,
        doc_ids_path: Path = config.DOC_IDS_PATH,
        corpus_path: Path = config.CORPUS_CACHE_PATH,
        model_name: str = config.DENSE_MODEL,
        corpus: dict | None = None,
        backend: str = config.DENSE_BACKEND,
    ) -> "DenseRetriever":
        index = faiss.read_index(str(index_path))
        index.hnsw.efSearch = config.HNSW_EF_SEARCH
        with open(doc_ids_path, "rb") as f:
            doc_ids = pickle.load(f)
        if corpus is None:
            with open(corpus_path, "rb") as f:
                corpus = pickle.load(f)
        if backend == "onnx":
            from retrieval.onnx_encoder import OnnxEncoder
            model = OnnxEncoder(config.ONNX_DENSE_DIR / "model.onnx")
        else:
            model = SentenceTransformer(model_name)
        return cls(index, doc_ids, corpus, model)

    def search(self, query: str, top_k: int = config.TOP_K) -> list[tuple[str, float, str]]:
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        # FAISS returns -1 for slots it can't fill (only happens if top_k > index size)
        return [
            (self.doc_ids[i], float(s), self.corpus[self.doc_ids[i]]["text"])
            for i, s in zip(indices[0], scores[0])
            if i >= 0
        ]
