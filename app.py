"""Self-Healing RAG — Streamlit demo app."""
import os

import streamlit as st

st.set_page_config(page_title="Self-Healing RAG", page_icon="🩹", layout="wide")

from src import config
from src.generation import LLMClient
from src.ingest import build_index, load_index
from src.pipeline import SelfHealingRAG
from src.retrieval import Retriever
from src.verification import Verifier

KIND_ICONS = {"info": "", "failure": "⚠️ ", "healing": "🩹 ", "success": "✅ "}


@st.cache_resource(show_spinner="Loading embedding + verification models (first run downloads ~1 GB)...")
def load_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    nli = CrossEncoder(config.NLI_MODEL)
    relevance = CrossEncoder(config.RELEVANCE_MODEL)
    return embedder, nli, relevance


@st.cache_resource(show_spinner="Building FAISS index...")
def get_index(_embedder):
    index, metadata = load_index()
    if index is None:
        index, metadata = build_index(_embedder)
    return index, metadata


def get_api_key() -> str | None:
    # Priority: user-entered key > env var > HF Spaces secret (also env var)
    key = st.session_state.get("user_api_key") or os.environ.get("GROQ_API_KEY") or ""
    return key.strip() or None


def render_log(log):
    lines = []
    for entry in log:
        icon = KIND_ICONS.get(entry.kind, "")
        lines.append(f"[{entry.step}] {icon}{entry.message}")
    st.code("\n".join(lines), language=None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🩹 Self-Healing RAG")
    st.markdown(
        "A RAG pipeline that **detects its own failures** and heals them:\n"
        "- Low retrieval similarity → **query rewriting**\n"
        "- Irrelevant chunks → **HyDE**\n"
        "- Ungrounded claims → **claim-targeted re-retrieval**\n"
        "- Empty answers → **retrieval expansion**"
    )
    st.divider()
    if not os.environ.get("GROQ_API_KEY"):
        st.text_input("Groq API key", type="password", key="user_api_key",
                      help="Free at console.groq.com")
    st.caption(f"Corpus: `data/docs/` • Embeddings: MiniLM-L6 • NLI: DeBERTa-v3 • LLM: {config.GROQ_MODEL}")
    st.divider()
    st.markdown("**Try these:**")
    st.markdown(
        "- *No healing:* “How does Raft elect a leader?”\n"
        "- *Query rewrite:* “why my cluster keep picking new boss?”\n"
        "- *Expansion:* “Compare HNSW and IVF and how Raft snapshots work”"
    )

# ---------------------------------------------------------------- main
st.header("Ask the corpus")

api_key = get_api_key()
if not api_key:
    st.info("Enter a Groq API key in the sidebar to start (free tier: console.groq.com).")
    st.stop()

embedder, nli, relevance = load_models()
index, metadata = get_index(embedder)
retriever = Retriever(embedder, index, metadata)
pipeline = SelfHealingRAG(retriever, LLMClient(api_key=api_key), Verifier(nli, relevance))

question = st.text_input("Question", placeholder="How does Raft elect a leader?")

if question:
    with st.spinner("Running self-healing pipeline..."):
        try:
            result = pipeline.answer(question)
        except Exception as e:
            msg = str(e)
            if "invalid_api_key" in msg or "401" in msg:
                st.error("Groq rejected the API key (401). Check it: keys start with `gsk_`, "
                         "no spaces, created at console.groq.com → API Keys.")
            elif "rate limit" in msg.lower() or "429" in msg:
                st.error("Groq rate limit hit — wait a few seconds and retry.")
            else:
                st.error(f"Pipeline error: {msg}")
            st.stop()

    col_answer, col_log = st.columns([3, 2])

    with col_answer:
        st.subheader("Answer")
        st.markdown(result.answer)

        if result.healed:
            st.warning(f"🩹 Self-healed: {result.healing_steps} healing step(s) applied")
        else:
            st.success("✅ No healing needed")

        if result.groundedness and result.groundedness["claims"]:
            with st.expander("Groundedness verification", expanded=False):
                for r in result.groundedness["claims"]:
                    icon = "✅" if r["grounded"] else "⚠️"
                    st.markdown(f"{icon} `{r['score']:.2f}` — {r['claim']}")

        with st.expander("Retrieved context", expanded=False):
            for c in result.chunks:
                st.markdown(f"**{c['source']}** (similarity {c['score']:.2f})")
                st.caption(c["text"])

    with col_log:
        st.subheader("Healing log")
        render_log(result.log)
