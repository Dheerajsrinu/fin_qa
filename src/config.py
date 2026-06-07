from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "datasets" / "fiqa"
INDEX_DIR = ROOT / "indexes"
RESULTS_DIR = ROOT / "results"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
DENSE_INDEX_PATH = INDEX_DIR / "dense.faiss"
CORPUS_CACHE_PATH = INDEX_DIR / "corpus.pkl"
DOC_IDS_PATH = INDEX_DIR / "doc_ids.pkl"

TOP_K = 10
RRF_K = 60
MAX_SEQ_LEN = 512
DENSE_BATCH_SIZE = 64
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50
