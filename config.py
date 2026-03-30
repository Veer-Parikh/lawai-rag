"""
config.py — Central configuration for Bharat Law RAG chatbot.
All tuneable parameters live here; pulled from environment via python-dotenv.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# ─── Pinecone ─────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX", "bharat-law")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSION: int = 768  # BGE-base-en output size

# Namespaces
NAMESPACE_STATUTES_CRIMINAL: str = "statutes-criminal"
NAMESPACE_STATUTES_CIVIL: str = "statutes-civil"
NAMESPACE_STATUTES_ECONOMIC: str = "statutes-economic"
NAMESPACE_STATUTES_FAMILY: str = "statutes-family"
NAMESPACE_STATUTES_CONSTITUTIONAL: str = "statutes-constitutional"
NAMESPACE_JUDGMENTS: str = "judgments-all"
NAMESPACE_MAPPINGS: str = "mappings"
NAMESPACE_MISC: str = "misc"

ALL_NAMESPACES: list[str] = [
    NAMESPACE_STATUTES_CRIMINAL,
    NAMESPACE_STATUTES_CIVIL,
    NAMESPACE_STATUTES_ECONOMIC,
    NAMESPACE_STATUTES_FAMILY,
    NAMESPACE_STATUTES_CONSTITUTIONAL,
    NAMESPACE_JUDGMENTS,
    NAMESPACE_MAPPINGS,
    NAMESPACE_MISC,
]

DOMAIN_TO_NAMESPACES: dict[str, list[str]] = {
    "criminal": [NAMESPACE_STATUTES_CRIMINAL, NAMESPACE_JUDGMENTS, NAMESPACE_MAPPINGS],
    "civil": [NAMESPACE_STATUTES_CIVIL, NAMESPACE_JUDGMENTS],
    "economic": [NAMESPACE_STATUTES_ECONOMIC, NAMESPACE_JUDGMENTS],
    "family": [NAMESPACE_STATUTES_FAMILY, NAMESPACE_JUDGMENTS],
    "constitutional": [NAMESPACE_STATUTES_CONSTITUTIONAL, NAMESPACE_JUDGMENTS],
    "misc": [NAMESPACE_MISC, NAMESPACE_JUDGMENTS],
    "general": ALL_NAMESPACES,
}

# ─── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en"
EMBEDDING_BATCH_SIZE: int = 32

# ─── Chunking ─────────────────────────────────────────────────────────────────
STATUTE_MAX_TOKENS: int = 512
JUDGMENT_MAX_TOKENS: int = 400
JUDGMENT_OVERLAP_TOKENS: int = 50

# ─── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K_FETCH: int = 5       # Chunks fetched per namespace before reranking
RETRIEVAL_TOP_K_FINAL: int = 5        # Chunks passed to LLM after reranking
RETRIEVAL_MAX_CROSS_REFS: int = 2     # Max cross-reference chunks auto-appended

# Confidence scoring thresholds
CONFIDENCE_HIGH_THRESHOLD: float = 0.80
CONFIDENCE_LOW_THRESHOLD: float = 0.65

# ─── Reranker ─────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"

# ─── Gemini ───────────────────────────────────────────────────────────────────
GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
GEMINI_TEMPERATURE: float = 0.1
GEMINI_TOP_P: float = 0.95
GEMINI_MAX_OUTPUT_TOKENS: int = 2048

# ─── Domain Router ────────────────────────────────────────────────────────────
DOMAIN_LRU_CACHE_SIZE: int = 512      # Max cached domain classifications

# ─── Session Store ────────────────────────────────────────────────────────────
SESSION_MAX_TURNS: int = 10           # Max turns kept per session in memory

# ─── Ingestion ────────────────────────────────────────────────────────────────
PINECONE_UPSERT_BATCH_SIZE: int = 100
DATA_DIR: str = os.getenv("DATA_DIR", "data")

# ─── Observability ────────────────────────────────────────────────────────────
EVAL_LOG_PATH: str = os.getenv("EVAL_LOG_PATH", "observability/eval.jsonl")

# ─── API ──────────────────────────────────────────────────────────────────────
API_RATE_LIMIT: str = "30/minute"
ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "")
CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
