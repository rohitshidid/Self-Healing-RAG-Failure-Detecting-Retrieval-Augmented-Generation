---
title: Self-Healing RAG
emoji: 🩹
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: "1.58.0"
app_file: app.py
pinned: false
---

# 🩹 Self-Healing RAG

A Retrieval-Augmented Generation pipeline that **detects its own failures and automatically heals them** before returning a bad answer. A normal RAG system retrieves chunks, generates, and returns — whether or not retrieval actually worked. This one doesn't fail silently.

> 📖 **Full guide** — simple + technical explanations, testing walkthrough, eval results, demo script: [PROJECT_GUIDE.md](PROJECT_GUIDE.md)

## Failure modes → healing strategies

| Failure detected | How it's detected | Healing strategy |
|---|---|---|
| Low retrieval similarity | Avg cosine similarity < threshold | **Query rewriting** — LLM rephrases the query, retry retrieval |
| Wrong chunks retrieved | NLI relevance score of chunks vs. question < threshold | **HyDE** — generate a hypothetical answer, embed *that*, retrieve with it |
| Ungrounded claims in answer | Per-sentence NLI entailment vs. retrieved chunks | **Claim-targeted re-retrieval** — search for the failing claim directly, regenerate |
| Empty / "I don't know" answer | Refusal pattern match despite chunks existing | **Retrieval expansion** — widen search to more chunks, retry generation |

Every step is written to a **healing log** rendered live next to the answer, so you can watch the system catch and fix itself.

## Stack (all free)

- **Streamlit** UI, hosted on **Hugging Face Spaces**
- **FAISS** flat inner-product index (file-based, no external DB)
- **sentence-transformers/all-MiniLM-L6-v2** embeddings (CPU)
- **cross-encoder/nli-deberta-v3-base** for relevance + groundedness verification (CPU)
- **Groq API** (free tier) for generation, query rewriting, and HyDE

## Project structure

```
app.py                  Streamlit app (answer + healing-log panel)
src/
  config.py             All thresholds, models, paths
  ingest.py             Load docs → chunk → embed → FAISS index
  retrieval.py          Vector search (by query or by embedding, for HyDE)
  generation.py         Groq LLM client: answer / rewrite / HyDE / claim-refine prompts
  verification.py       NLI verifier: chunk relevance + per-claim groundedness
  healing.py            The four healing strategies
  pipeline.py           Orchestrator: detect → heal → verify → log
data/docs/              The corpus (.txt/.md) — swap in your own docs
eval/
  eval_set.json         20-question eval set (incl. adversarial casual phrasings)
  run_eval.py           Self-healing vs. baseline-RAG comparison
tests/
  test_pipeline.py      8 unit tests — every failure→healing route, no API needed
  smoke_test.py         Integration test with real models (LLM faked without API key)
```

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Get a free API key at https://console.groq.com
export GROQ_API_KEY=gsk_...

streamlit run app.py
```

First launch downloads ~800 MB of models (MiniLM + DeBERTa NLI) and builds the FAISS index; subsequent launches are fast. You can also paste the Groq key into the sidebar instead of setting the env var.

## Test

```bash
pytest tests/test_pipeline.py -v      # unit tests, no downloads, no API key
python -m tests.smoke_test            # real embeddings + NLI (LLM faked w/o key)
GROQ_API_KEY=... python -m eval.run_eval   # full eval vs. baseline
```

## Demo queries

- **No healing:** "How does Raft elect a leader?"
- **Query-rewrite healing:** "why my cluster keep picking new boss all the time"
- **Expansion:** "Compare HNSW and IVF and how Raft snapshots work"
- **Honest refusal:** "What is the capital of France?" (not in corpus)

## Use your own documents

Drop `.txt`/`.md` files into `data/docs/`, delete `data/index/`, and restart. The index rebuilds automatically.

## Tuning

All knobs live in `src/config.py`: similarity/relevance/groundedness thresholds, chunk size, top-k, max healing attempts.
