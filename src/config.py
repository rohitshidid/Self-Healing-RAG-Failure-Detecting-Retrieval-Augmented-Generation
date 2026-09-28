"""Central configuration for the Self-Healing RAG pipeline."""
import os

# --- Models ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
RELEVANCE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
# Tried in order if the primary model is retired or not available on the key's tier
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]

# --- Retrieval ---
CHUNK_SIZE = 500            # characters per chunk
CHUNK_OVERLAP = 100         # character overlap between chunks
TOP_K = 4                   # chunks retrieved per query
EXPANDED_TOP_K = 8          # chunks retrieved during retrieval-expansion healing

# --- Failure-detection thresholds ---
SIMILARITY_THRESHOLD = 0.40   # avg cosine similarity below this = retrieval failure
RELEVANCE_THRESHOLD = -8.0    # max ms-marco cross-encoder logit below this = wrong chunks
                              # (calibrated: on-topic queries score >= -4, off-topic ~ -11)
GROUNDEDNESS_THRESHOLD = 0.50 # NLI entailment prob below this = ungrounded claim

# --- Healing ---
MAX_HEALING_ATTEMPTS = 3    # hard cap so the pipeline always terminates

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "docs")
INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "index")
