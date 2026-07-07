# Self-Healing RAG — The Complete Guide

*Everything about this project in one document: what it is, why it exists, how it works (simple and technical), how to run and test it, what to expect, and how to talk about it.*

---

## Table of Contents

1. [The Simple Explanation (no background needed)](#1-the-simple-explanation)
2. [The Problem This Solves](#2-the-problem-this-solves)
3. [How It Works — Simple Version](#3-how-it-works--simple-version)
4. [How It Works — Technical Version](#4-how-it-works--technical-version)
5. [The Tech Stack and Why Each Piece Was Chosen](#5-the-tech-stack)
6. [Project Structure — What Every File Does](#6-project-structure)
7. [How to Run It](#7-how-to-run-it)
8. [How to Test It — and Exactly What to Expect](#8-how-to-test-it)
9. [Evaluation Results (Real Numbers)](#9-evaluation-results)
10. [Live Demo Script (for interviews)](#10-live-demo-script)
11. [Design Decisions Worth Knowing](#11-design-decisions)
12. [Known Limitations and Future Work](#12-known-limitations)
13. [Glossary](#13-glossary)

---

## 1. The Simple Explanation

Imagine a librarian who answers questions by first fetching relevant books, then reading you an answer based on them.

A **normal** AI librarian has a flaw: even when they grab the *wrong* books — or misread them — they still confidently give you an answer. They never check their own work. This is how most Retrieval-Augmented Generation (RAG) systems behave today: they fail *silently*.

**This project builds a librarian that checks their own work.** After fetching books, it asks itself: *"Are these actually the right books?"* After composing an answer, it asks: *"Is every sentence I just said actually supported by these books?"* And when the answer to either question is *no*, it doesn't give up or bluff — it **tries a different strategy**: rephrases the search, imagines what the right book would look like and searches for that, fetches more books, or hunts down evidence for the specific sentence it couldn't back up.

Every check and every retry is displayed in a live "healing log" next to the answer, so you can literally watch the system catch and fix its own mistakes in real time.

That's it. **A question-answering AI that detects its own failures and recovers from them automatically** — instead of failing silently like almost every RAG demo ever built.

---

## 2. The Problem This Solves

RAG (Retrieval-Augmented Generation) is the standard way to make an LLM answer questions about *your* documents: chop documents into chunks, store them in a searchable index, fetch the most relevant chunks for each question, and have the LLM answer using only those chunks.

The standard pipeline has four silent failure modes:

| # | Failure | What the user experiences |
|---|---------|--------------------------|
| 1 | **Query mismatch** — user phrases the question casually ("why my cluster keep picking new boss?") but documents use formal language ("leader election") | Retrieval returns junk; LLM answers from junk or hallucinates |
| 2 | **Wrong chunks** — retrieval finds text that's *superficially* similar but actually about something else | Confident answer about the wrong thing |
| 3 | **Hallucination** — the LLM adds claims that aren't in the retrieved text at all | Plausible-sounding falsehoods, presented with full confidence |
| 4 | **Over-refusal** — the LLM says "I don't know" even though the answer *was* retrievable | Unnecessary dead end |

Failures 3 and 4 are **LLM-specific failure modes**; failures 1 and 2 are **retrieval failure modes**. A production-shaped system needs to handle both layers. Most portfolio projects build only the happy path — this project is specifically the *failure-handling layer*.

---

## 3. How It Works — Simple Version

Every question goes through a five-stage assembly line. At two checkpoints the system grades its own work, and each possible failure has a dedicated repair strategy:

```
Question
   │
   ▼
[1] SEARCH the document index for relevant chunks
   │
   ▼
[2] CHECKPOINT: Did the search work?
   ├── Similarity too low?  →  🩹 Ask the LLM to REPHRASE the question, search again
   ├── Chunks off-topic?    →  🩹 HyDE: LLM writes a FAKE ANSWER, search using that
   │                            (answers resemble answers more than questions do)
   ▼
[3] GENERATE an answer using only the retrieved chunks
   │
   ▼
[4] CHECKPOINT: Is the answer trustworthy?
   ├── Answer says "I don't know" but chunks exist?
   │        →  🩹 Fetch MORE chunks (wider net), try generating again
   ├── Answer contains a sentence NOT supported by the chunks?
   │        →  🩹 Search for evidence for THAT SENTENCE specifically,
   │            add it to the context, regenerate the answer
   ▼
[5] RETURN the answer + the full log of everything that happened
```

Key properties:

- **Fact-checking is per-sentence.** The answer is split into individual claims, and each one must be backed by the retrieved text (verified by a separate AI model trained to judge "does text A support statement B?").
- **It always terminates.** Healing is capped at 3 attempts, so a hopeless question can't loop forever.
- **It's honest.** If the corpus genuinely doesn't contain the answer ("What's the capital of France?" asked of a distributed-systems corpus), it refuses — and the log shows it tried.
- **Everything is visible.** The healing log shows each detection and repair, step by step.

---

## 4. How It Works — Technical Version

### 4.1 Ingestion (`src/ingest.py`)

- Documents (`.txt`/`.md` in `data/docs/`) are chunked with a **sliding character window**: 500 chars, 100-char overlap, cut points snapped to sentence boundaries, window starts snapped to word boundaries.
- Each chunk is embedded with **`all-MiniLM-L6-v2`** (384-dim sentence-transformer, CPU-friendly), **L2-normalized**, and stored in a **FAISS `IndexFlatIP`** — with normalized vectors, inner product ≡ cosine similarity. Exact (non-approximate) search; fine at this corpus scale.
- Index + chunk metadata persist to `data/index/` and rebuild automatically if missing.

### 4.2 Retrieval (`src/retrieval.py`)

- Query → same embedder → normalized vector → top-k (default **k=4**) FAISS search.
- Two entry points: `retrieve(query)` for text, and `retrieve_by_embedding(vector)` — the latter exists specifically for HyDE, which retrieves with the embedding of a *generated document* rather than the question.

### 4.3 Failure Detection — the interesting part

**Signal 1 — retrieval confidence** (`pipeline.py` stage 1):
Average cosine similarity of the top-k chunks. Below **0.40** → the query, as phrased, doesn't live near anything in the corpus's embedding space. Classic cause: vocabulary mismatch between casual questions and formal documents.

**Signal 2 — semantic relevance** (`verification.py`):
Cosine similarity can be *high* while chunks are still about the wrong thing (embeddings compress a lot). So chunks are re-scored by a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`), which reads the query and chunk *together* — far more accurate than comparing two independent embeddings. If the **best** chunk's logit is below **-8.0**, the retrieved set is judged off-topic. (This threshold was calibrated empirically on this corpus: on-topic queries score ≥ -4, off-topic ≈ -11.)

**Signal 3 — refusal detection** (`generation.py::is_non_answer`):
Pattern-match for "I don't know" / empty answers. Refusal *despite* decent chunks usually means the relevant fact is just outside the retrieved window — fixable by widening retrieval.

**Signal 4 — groundedness / hallucination detection** (`verification.py`):
The core LLM-failure check. The answer is split into sentence-level claims. Each claim becomes an NLI **hypothesis**; premises are the retrieved chunks **plus 2-sentence sliding windows within each chunk**. The NLI cross-encoder (`nli-deberta-v3-base`) outputs P(entailment) per (premise, claim) pair; a claim is grounded if its **max** entailment ≥ **0.50**.

> Why the sentence windows? Discovered during testing: DeBERTa NLI dramatically under-scores evidence buried mid-way through a long premise — a claim that appeared **verbatim** inside a 500-char chunk scored just 0.03 entailment against that chunk, but 0.98+ against the right 2-sentence window. Chunk-level premises alone make the hallucination detector unreliable; window-level premises fix it while chunk-level premises still catch claims whose evidence spans many sentences.

### 4.4 Healing Strategies (`src/healing.py`) — one per failure

| Failure signal | Strategy | Mechanism |
|---|---|---|
| Low similarity (S1) | **Query rewriting** | LLM rephrases with formal/domain vocabulary → retry retrieval. Adopted only if similarity actually improves. The relevance check afterward runs against the *rewritten* query (the query that produced the chunks) — scoring new chunks against the old casual query caused false alarms. |
| Off-topic chunks (S2) | **HyDE** (Hypothetical Document Embeddings) | LLM writes a short hypothetical *answer* paragraph; its embedding retrieves the chunks. Works because answer-shaped text lands nearer to answer-shaped documents than question-shaped text does. |
| Refusal despite chunks (S3) | **Retrieval expansion** | Re-retrieve with k=8 instead of 4 → regenerate. If it still refuses, the refusal is accepted as honest. |
| Ungrounded claim (S4) | **Claim-targeted re-retrieval** | LLM converts the *failing claim* into a search query → results merge with existing context (deduplicated, capped at 8) → regenerate → re-verify. Surgical: fetches evidence for exactly the sentence that failed. |

Global cap: **3 healing attempts per query** (`MAX_HEALING_ATTEMPTS`) — guarantees termination.

### 4.5 Generation (`src/generation.py`)

Groq API, `llama-3.3-70b-versatile`, temperature 0.2 for answers. The system prompt forbids outside knowledge, mandates the exact refusal string, and — importantly — **forbids hedging ("may/might/could") and meta-commentary about the context**. This isn't stylistic: hedged or meta sentences are unverifiable as NLI hypotheses, so the generator must produce claims the verifier can actually check. *Generator and verifier must be aligned.* Rate limits (429) are retried with exponential backoff (5s/10s/20s).

### 4.6 Orchestration (`src/pipeline.py`)

`SelfHealingRAG.answer(question)` runs the stages, tracks the *active query* (original or adopted rewrite), appends every event to a typed log (`info` / `failure` / `healing` / `success`), and returns a `RAGResult`: answer, final chunks, per-claim groundedness, healing-step count, full log. The Streamlit app (`app.py`) is a thin view over this object — all logic lives in the pipeline, which is why it's unit-testable without any UI.

---

## 5. The Tech Stack

| Component | Choice | Why |
|---|---|---|
| Vector store | **FAISS** (flat IP index) | File-based, zero external services, exact search, free |
| Embeddings | **all-MiniLM-L6-v2** | 384-dim, fast on CPU, no API cost |
| Relevance check | **ms-marco-MiniLM-L-6-v2** cross-encoder | Purpose-trained for query–passage relevance; tiny (~80 MB) |
| Hallucination check | **nli-deberta-v3-base** cross-encoder | Purpose-trained NLI (entailment); local, no API cost |
| LLM | **Groq API**, llama-3.3-70b | Free tier, very fast inference (good for live demos) |
| UI | **Streamlit** | Fastest path to a clean demo UI |
| Hosting | **Hugging Face Spaces** (free CPU) | Permanent public URL, no credit card, never expires |
| Testing | **pytest** + deterministic fakes | Every failure→healing route testable offline in <1s |

Total cost to run forever: **$0**. Only the LLM needs a network call; both verifier models run locally on CPU.

---

## 6. Project Structure

```
app.py                    Streamlit UI — answer panel + live healing log + key handling
src/
  config.py               Every tunable: model names, thresholds, chunk size, k, healing cap
  ingest.py               Load docs → chunk → embed → build/persist FAISS index
  retrieval.py            Top-k search by query text or by raw embedding (HyDE)
  generation.py           Groq client: 4 prompts (answer/rewrite/HyDE/claim-refine),
                          refusal detector, 429 retry
  verification.py         Cross-encoder relevance scoring + per-claim NLI groundedness
  healing.py              The 4 healing strategies (pure functions: state in → chunks out)
  pipeline.py             SelfHealingRAG orchestrator + typed healing log
data/
  docs/                   Corpus: 4 docs (Raft consensus ×2, vector DBs, RAG systems)
  index/                  Auto-generated FAISS index + metadata (safe to delete)
eval/
  eval_set.json           20 questions: formal, casual/adversarial, multi-topic, 4 traps
  run_eval.py             Head-to-head: self-healing vs baseline RAG, writes eval_results.json
  eval_results.json       Full per-question results from the real run
tests/
  fakes.py                Deterministic FakeEmbedder / FakeLLM / FakeVerifier
  test_pipeline.py        8 unit tests — every detection + healing route, offline
  smoke_test.py           Integration test, real models (real LLM if key set, fake otherwise)
README.md                 Short version of this doc + HF Spaces config frontmatter
DEPLOY.md                 Step-by-step permanent deployment guide
PROJECT_GUIDE.md          This document
```

**Swap in your own documents:** drop `.txt`/`.md` files into `data/docs/`, delete `data/index/`, restart. Index rebuilds automatically.

---

## 7. How to Run It

```bash
cd "/Users/rohitshidid/Documents/AntiGravity/RAG Project"
source .venv/bin/activate                 # venv already exists with all deps
export GROQ_API_KEY=gsk_...               # free key: console.groq.com → API Keys
streamlit run app.py                      # opens http://localhost:8501
```

- First-ever launch downloads ~1 GB of models (one-time; already cached on this machine).
- No env var? Paste the key into the sidebar field instead.
- Fresh machine? `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` first.

---

## 8. How to Test It

### 8.1 Unit tests — instant, no API key, no downloads

```bash
.venv/bin/python -m pytest tests/test_pipeline.py -v
```

**Expect:** `8 passed` in under a second. What they prove:

| Test | Proves |
|---|---|
| `test_chunking_covers_text_with_overlap` | Chunker respects size limits, produces real substrings |
| `test_is_non_answer` | Refusal detector catches variants, passes real answers |
| `test_no_healing_when_everything_ok` | Healthy path adds zero healing steps |
| `test_low_similarity_triggers_query_rewrite` | Low similarity → rewrite path fires, similarity improves |
| `test_irrelevant_chunks_trigger_hyde` | Bad relevance → HyDE path fires |
| `test_empty_answer_triggers_retrieval_expansion` | Refusal → expansion fires, second attempt answers |
| `test_ungrounded_claim_triggers_reretrieval` | Failed claim → re-retrieval fires, re-check passes |
| `test_healing_attempts_are_capped` | Hopeless input still terminates, ≤ 3 heals |

These use scripted fakes (deterministic embedder/LLM/verifier), so they test the *routing logic* — which failure triggers which strategy — in isolation.

### 8.2 Integration smoke test — real models

```bash
GROQ_API_KEY=gsk_... .venv/bin/python -m tests.smoke_test
```

**Expect** (~1–2 min on CPU), two traces:

Query 1, `How does Raft elect a leader?` — the healthy path:
```
[1] Initial query received
[2] Retrieval attempted → avg similarity: 0.49 ✓
[3] Generating answer...
[4] Groundedness check: 3/3 claims verified ✓
[5] Answer returned (no healing needed)
```

Query 2, `why my cluster keep picking new boss all the time` — deliberately casual phrasing:
```
[1] Initial query received
[2] Retrieval attempted → avg similarity: 0.32 (below threshold)
[3] ⚠️ Failure detected: low retrieval confidence
[4] 🩹 Healing: rewriting query via LLM...
[5] Retry retrieval — rewrote query to: "What are the underlying causes of
    frequent leader election..." → avg similarity: 0.52 ✓
[6] Generating answer...
[7] Groundedness check: 3/3 claims verified ✓
[8] Answer returned after 1 healing step
```

Exact similarity numbers and rewritten wording vary slightly per run (LLM sampling); the *shape* — failure detected, heal applied, verified answer — is stable. Without a key it runs with a fake LLM; retrieval and verification are still real.

### 8.3 In the app — what to type and what you'll see

| Type this | Expect |
|---|---|
| `How does Raft elect a leader?` | Green "No healing needed", 3/3 claims verified, log ~5 lines |
| `why my cluster keep picking new boss all the time` | ⚠️ low-similarity failure → rewrite heal → verified answer, yellow "Self-healed" badge |
| `how do those vector thingies get squished to save memory` | Casual→formal heal, answer about Product Quantization |
| `Compare HNSW and IVF and how Raft snapshots work` | Multi-topic; often triggers expansion or claim re-retrieval |
| `What is the capital of France?` | Honest refusal — corpus doesn't cover it; log shows healing was *attempted* |

Also in the UI: expanders showing per-claim verification scores and the retrieved chunks with similarities. First query after launch is slow (~30–60 s, models loading into memory); subsequent queries take a few seconds.

### 8.4 Full evaluation — the resume numbers

```bash
GROQ_API_KEY=gsk_... .venv/bin/python -m eval.run_eval
```

Runs all 20 questions through both a baseline RAG (retrieve → generate → return, no checks) and the self-healing pipeline. ~10–15 min on CPU. Writes `eval/eval_results.json`.

---

## 9. Evaluation Results

Real run, 2026-07-06, 20-question eval set (16 answerable, 4 traps):

| Metric (16 answerable) | Baseline RAG | Self-Healing RAG |
|---|---|---|
| Answered | 16/16 | 16/16 |
| **Fully grounded** (every claim verified) | **11/16 (69%)** | **13/16 (81%)** |
| Ungrounded answers | 5 | 3 (**−40%**) |
| Trap questions correctly refused | 4/4 | 4/4 |
| Healing steps triggered | — | 20 |

Where healing won: casual phrasings (`"vector thingies get squished"` — baseline ungrounded, healed grounded) and multi-topic queries (`"Compare HNSW with Raft snapshots"` — same pattern). Honest imperfections, visible in `eval_results.json`: one question came out *less* grounded after healing (LLM sampling noise), one stayed ungrounded after its heal. Real systems have real numbers.

---

## 10. Live Demo Script

90-second version:

1. **Healthy path** — ask `How does Raft elect a leader?` → point at the log: retrieval good, every claim fact-checked, no healing. *"Standard RAG, plus verification."*
2. **The money shot** — ask `why my cluster keep picking new boss all the time` → walk the log: *"Similarity 0.32 — the system knows the search failed before answering. Watch: it rewrites the query itself... 0.52, passes. And every sentence of the answer is NLI-verified against the sources."*
3. **Honesty** — ask `What is the capital of France?` → *"Not in the corpus. It tried healing, couldn't find grounding, and refuses instead of hallucinating. This is the difference between a demo and something production-shaped."*
4. Close with the eval numbers: **69% → 81% grounded, 40% fewer ungrounded answers.**

Talking-point pairing with the Raft project: *"I build systems that detect their own failures and recover automatically — at the infrastructure layer (Raft) and at the AI layer (this)."*

---

## 11. Design Decisions

Things that were discovered/decided during the build — good interview material:

1. **NLI is the wrong tool for relevance.** First version used the NLI model for "is this chunk relevant?" — it over-fired constantly (correct chunks scored 0.25 against a "this passage is relevant" hypothesis). Swapped to a purpose-trained query–passage cross-encoder and calibrated the threshold empirically. *Lesson: use models for what they were trained for.*
2. **NLI misses evidence buried in long premises.** A claim appearing verbatim mid-chunk scored 0.03 entailment. Fixed with 2-sentence sliding-window premises alongside full chunks. *Lesson: verify your verifier.*
3. **Generator and verifier must be aligned.** The LLM's hedges ("may cause...") and meta-commentary ("the context discusses...") are unverifiable as NLI hypotheses — so the answer prompt forbids them. The prompt is part of the verification system.
4. **Score relevance against the query that produced the chunks.** After a rewrite heal, checking new chunks against the *original* casual query caused false HyDE triggers.
5. **Adopt heals only on measured improvement** (similarity/relevance must actually increase), and **cap total heals** — self-healing systems need anti-flapping guards, same as infrastructure.
6. **All logic in the pipeline, none in the UI** — which is what makes 8 offline unit tests possible.

---

## 12. Known Limitations

- **Non-determinism:** LLM sampling means a healed regeneration can occasionally be worse (seen once in eval). Mitigation would be best-of-n with groundedness scoring.
- **No prompt-injection defense:** retrieved chunks go into the prompt untreated; a malicious document could steer the LLM. Natural next feature.
- **Refusal detection is pattern-based:** an exotic refusal phrasing could slip past.
- **Thresholds are corpus-calibrated:** a very different corpus (legal, medical) would need recalibration of the −8.0 relevance and 0.40 similarity thresholds.
- **Small eval set (20):** numbers are indicative, not statistically tight.
- **Sequential healing:** strategies run one at a time; parallel multi-query retrieval would be faster but costs more LLM calls.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — LLM answers using documents fetched at question time |
| **Chunk** | A ~500-character slice of a document, the unit of retrieval |
| **Embedding** | A vector (list of numbers) encoding text meaning; similar meanings → nearby vectors |
| **Cosine similarity** | How close two embeddings point (1.0 identical, ~0 unrelated) |
| **FAISS** | Meta's library for fast vector similarity search |
| **Cross-encoder** | Model that reads two texts *together* to score their relationship — slower but far more accurate than comparing embeddings |
| **NLI** | Natural Language Inference — does premise text entail (support) a hypothesis statement? |
| **Entailment** | The premise being true guarantees the hypothesis is true |
| **Groundedness** | Every claim in an answer is entailed by retrieved evidence |
| **Hallucination** | LLM output not supported by its source context |
| **HyDE** | Hypothetical Document Embeddings — retrieve using the embedding of a generated hypothetical answer instead of the question |
| **Top-k** | Number of chunks fetched per search (4 normal, 8 expanded) |
| **Healing step** | One detected failure + one applied repair strategy |
