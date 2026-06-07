"""
ONNX query encoder — a drop-in for SentenceTransformer.encode().

Replicates the SentenceTransformer pipeline exactly so embeddings match the
PyTorch ones the FAISS index was built with:
    tokenize -> transformer (ONNX) -> pooling (cls|mean) -> L2 normalize.

Pooling differs per model:
    - bge-small / bge-base : CLS pooling   (last_hidden_state[:, 0])
    - all-mpnet / all-MiniLM: MEAN pooling (mask-weighted average)
so the pooling mode is a constructor arg.

The set of model inputs (whether token_type_ids exists) is read straight from
the ONNX graph, so a BERT graph (input_ids/attention_mask/token_type_ids) and an
MPNet graph (input_ids/attention_mask only) both work without extra flags.

Only the query side is encoded at serve time (one short string per search), so
this is tuned for low single-query latency, not bulk throughput.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import config


ONNX_DIR = config.INDEX_DIR / "bge_small_onnx"


class OnnxEncoder:
    def __init__(
        self,
        onnx_path: Path = ONNX_DIR / "model.onnx",
        tokenizer_dir: Path | None = None,
        pooling: str = "cls",
        max_seq_len: int = config.MAX_SEQ_LEN,
        intra_op_threads: int | None = None,
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.pooling = pooling
        self.max_seq_len = max_seq_len

        tok_dir = tokenizer_dir if tokenizer_dir is not None else Path(onnx_path).parent
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is not None:
            opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        # Which inputs does this graph actually take? (BERT has token_type_ids, MPNet doesn't)
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(
        self,
        sentences: list[str] | str,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,   # accepted for signature parity; always numpy
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        tok = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="np",
        )
        inputs = {
            "input_ids":      tok["input_ids"].astype(np.int64),
            "attention_mask": tok["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in self._input_names:
            tt = tok.get("token_type_ids")
            if tt is None:
                tt = np.zeros_like(tok["input_ids"])
            inputs["token_type_ids"] = tt.astype(np.int64)

        last_hidden = self.session.run(["last_hidden_state"], inputs)[0]   # (b, seq, h)

        if self.pooling == "cls":
            pooled = last_hidden[:, 0]
        else:  # mean pooling, mask-weighted (matches SentenceTransformer)
            mask = inputs["attention_mask"][:, :, None].astype(np.float32)  # (b, seq, 1)
            summed = (last_hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = summed / counts

        if normalize_embeddings:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)

        return pooled.astype(np.float32)
