"""Self-Healing RAG pipeline orchestrator.

Flow per query:
  1. Retrieve chunks.
  2. Detect retrieval failures (low similarity -> query rewrite; irrelevant chunks -> HyDE).
  3. Generate answer.
  4. Detect generation failures (empty/'I don't know' -> retrieval expansion;
     ungrounded claims -> claim-targeted re-retrieval + regenerate).
  5. Return answer + full healing log.

Every step appends a HealingLogEntry so the UI can render the pipeline trace.
"""
from dataclasses import dataclass, field

from src import config, healing
from src.generation import is_non_answer
from src.retrieval import avg_similarity


@dataclass
class LogEntry:
    step: int
    kind: str          # "info" | "failure" | "healing" | "success"
    message: str


@dataclass
class RAGResult:
    question: str
    answer: str
    chunks: list[dict]
    log: list[LogEntry]
    healing_steps: int
    groundedness: dict | None = None
    healed: bool = field(init=False)

    def __post_init__(self):
        self.healed = self.healing_steps > 0


class SelfHealingRAG:
    def __init__(self, retriever, llm, verifier):
        self.retriever = retriever
        self.llm = llm
        self.verifier = verifier

    # ------------------------------------------------------------------
    def answer(self, question: str) -> RAGResult:
        log: list[LogEntry] = []
        step = [0]
        heals = [0]

        def add(kind, msg):
            step[0] += 1
            log.append(LogEntry(step[0], kind, msg))

        add("info", "Initial query received")

        # ---------- Stage 1: retrieval + retrieval-failure healing ----------
        active_query = question  # the query that produced the current chunks
        chunks = self.retriever.retrieve(question)
        sim = avg_similarity(chunks)
        add("info", f"Retrieval attempted → avg similarity: {sim:.2f}"
                    + (" ✓" if sim >= config.SIMILARITY_THRESHOLD else " (below threshold)"))

        if sim < config.SIMILARITY_THRESHOLD and heals[0] < config.MAX_HEALING_ATTEMPTS:
            add("failure", "Failure detected: low retrieval confidence")
            heals[0] += 1
            add("healing", "Healing: rewriting query via LLM...")
            new_chunks, note, rewritten = healing.heal_low_similarity(
                self.llm, self.retriever, question)
            new_sim = avg_similarity(new_chunks)
            add("info", f"Retry retrieval — {note} → avg similarity: {new_sim:.2f}"
                        + (" ✓" if new_sim >= config.SIMILARITY_THRESHOLD else ""))
            if new_sim > sim:
                chunks, sim = new_chunks, new_sim
                active_query = rewritten

        # Relevance check (only worth running when similarity looked OK,
        # otherwise the rewrite above already addressed retrieval).
        if sim >= config.SIMILARITY_THRESHOLD and chunks:
            relevance = self.verifier.chunk_relevance(active_query, chunks)
            best_rel = max(relevance)
            if best_rel < config.RELEVANCE_THRESHOLD and heals[0] < config.MAX_HEALING_ATTEMPTS:
                add("failure", f"Failure detected: chunks semantically irrelevant "
                               f"(best relevance {best_rel:.2f})")
                heals[0] += 1
                add("healing", "Healing: HyDE — embedding a hypothetical answer...")
                new_chunks, note = healing.heal_wrong_chunks(self.llm, self.retriever, question)
                new_rel = self.verifier.chunk_relevance(question, new_chunks)
                new_best_rel = max(new_rel) if new_rel else float("-inf")
                add("info", f"Retry retrieval — {note} → best relevance: {new_best_rel:.2f}"
                            + (" ✓" if new_best_rel >= config.RELEVANCE_THRESHOLD else ""))
                if new_best_rel > best_rel:
                    chunks = new_chunks

        # ---------- Stage 2: generation + generation-failure healing ----------
        add("info", "Generating answer...")
        answer = self.llm.generate_answer(question, chunks)

        if is_non_answer(answer) and heals[0] < config.MAX_HEALING_ATTEMPTS:
            add("failure", "Failure detected: empty / 'I don't know' answer despite retrieved chunks")
            heals[0] += 1
            add("healing", "Healing: expanding retrieval and retrying generation...")
            chunks, note = healing.heal_empty_answer(self.retriever, question)
            add("info", f"Retry — {note}")
            answer = self.llm.generate_answer(question, chunks)
            if is_non_answer(answer):
                add("info", "Answer still unavailable — corpus likely lacks this information")

        # ---------- Stage 3: groundedness verification ----------
        groundedness = None
        if not is_non_answer(answer):
            groundedness = self.verifier.check_groundedness(answer, chunks)
            n = len(groundedness["claims"])
            ok = sum(r["grounded"] for r in groundedness["claims"])
            add("info", f"Groundedness check: {ok}/{n} claims verified"
                        + (" ✓" if groundedness["all_grounded"] else ""))

            if not groundedness["all_grounded"] and heals[0] < config.MAX_HEALING_ATTEMPTS:
                bad = next(r for r in groundedness["claims"] if not r["grounded"])
                add("failure", f'Failure detected: ungrounded claim — "{bad["claim"][:100]}"')
                heals[0] += 1
                add("healing", "Healing: re-retrieving evidence for the claim and regenerating...")
                chunks, note = healing.heal_ungrounded(
                    self.llm, self.retriever, question, bad["claim"], chunks)
                add("info", f"Retry — {note}")
                answer = self.llm.generate_answer(question, chunks)
                groundedness = self.verifier.check_groundedness(answer, chunks)
                n = len(groundedness["claims"])
                ok = sum(r["grounded"] for r in groundedness["claims"])
                add("info", f"Groundedness re-check: {ok}/{n} claims verified"
                            + (" ✓" if groundedness["all_grounded"] else ""))

        # ---------- Done ----------
        if heals[0] == 0:
            add("success", "Answer returned (no healing needed)")
        else:
            plural = "step" if heals[0] == 1 else "steps"
            add("success", f"Answer returned after {heals[0]} healing {plural}")

        return RAGResult(
            question=question,
            answer=answer,
            chunks=chunks,
            log=log,
            healing_steps=heals[0],
            groundedness=groundedness,
        )
