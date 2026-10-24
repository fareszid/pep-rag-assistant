"""Paths, model names and retrieval settings."""

import os
from pathlib import Path


def _project_root() -> Path:
    """Use PEPBOT_ROOT if set, otherwise walk up from the cwd (then this file) to the project."""
    if os.getenv("PEPBOT_ROOT"):
        return Path(os.environ["PEPBOT_ROOT"])
    here = Path(__file__).resolve()
    for folder in [Path.cwd(), *Path.cwd().parents, *here.parents]:
        if (folder / "pyproject.toml").exists() and (folder / "eval").is_dir():
            return folder
    return Path.cwd()


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "peps"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.parquet"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
EVAL_QUESTIONS = PROJECT_ROOT / "eval" / "questions.jsonl"
REPORTS_DIR = PROJECT_ROOT / "reports"

PEPS_TARBALL_URL = "https://codeload.github.com/python/peps/tar.gz/refs/heads/main"
PEP_URL = "https://peps.python.org/pep-{number:04d}/"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CLAUDE_MODEL = os.getenv("PEPBOT_MODEL", "claude-opus-5-5")

CHUNK_MAX_WORDS = 220
RRF_K = 60  # value from the RRF paper (Cormack et al. 2009)
CANDIDATES = 50  # candidates from each retriever before fusion / reranking
TOP_K = 6  # passages sent to Claude
