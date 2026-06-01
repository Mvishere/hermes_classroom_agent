Hermes Classroom Agent — Detailed Architecture and Operational Guide
=================================================================

This document explains how the Hermes Classroom Agent works, its components, data flow, runtime configuration, and techniques used. It's written for a developer or demo operator who needs to understand, run, and present the system.

1. Project Overview
-------------------
Hermes Classroom Agent is a local-first assistant designed to ingest Google Classroom data, extract educational topics, build a topic graph, track student knowledge, and answer student questions using a Retrieval-Augmented Generation (RAG) pipeline. The design prioritizes demo reliability, local operation (no cloud deps), and reproducible outputs.

Goals
- Local RAG-based QA grounded in course materials
- Robust topic extraction and canonicalization for a clean topic graph
- Lightweight, modular architecture that runs in a local venv
- Demonstration-ready outputs (evidence, confidence, clean formatting)

2. High-level Architecture
--------------------------
- Data ingestion: Google Classroom API + Drive attachments + Form parsing
- Storage: local JSON files under `data/` and per-item topic payloads in `data/topics/`
- Topic extraction: hybrid heuristic + LLM extraction -> TopicCleaner -> Normalizer -> Ontology mapping
- Topic graph: `topic_graph` module builds a weighted graph (nodes=concepts, edges=related/prereq)
- Knowledge tracking: per-student/per-course topic mastery state in `data/knowledge_state.json`
- RAG pipeline: embeddings (local SentenceTransformer / `models/minilm`) + retrieval index in memory + LLM answer generation via configured backend (Ollama preferred)
- CLI: `python -m chat.cli` for interactive and batch QA
- Scheduler & Polling: `main.py` runs a polling loop with Playwright browser for attachments and quizzes, rebuilds graph periodically

3. Key Modules and Files
------------------------
- `main.py`: Orchestrates startup, polling loop, scheduler, and wires the pipeline. Supports `MAIN_SAFE_MODE` for quick checks.
- `config.py`: Centralized configuration; reads `.env` (does not override shell envs) and exposes keys such as `LLM_BACKEND`, `LLM_MODEL_PATH`, `EMBEDDING_MODEL_PATH`, `RAG_*` and `TOPIC_*` settings.
- `chat/assistant.py`: Chat front-end that builds an in-memory retrieval index using `EmbeddingModel`, applies similarity scoring, and formats grounded responses.
- `chat/cli.py`: CLI interface to the assistant; supports batch mode for stress tests.
- `rag/embeddings.py`: Wrapper around the chosen sentence-transformer model (`models/minilm` by default).
- `rag/llm.py`: `LocalLLM` facade — loads the backend via `rag/model_loader.py` and provides `generate()`/`stream()`.
- `rag/model_loader.py`: Backend selection: `TransformersQwenBackend`, `LlamaCppQwenBackend`, `OllamaBackend` (Python client preferred; CLI fallback). Use `LLM_BACKEND=ollama` and `LLM_MODEL_PATH=qwen2.5:7b` to target Qwen via Ollama.
- `topic_graph/`:
  - `extractor.py`: `SemanticTopicExtractor` (hybrid heuristic + LLM) — now uses strict JSON prompt for LLM topic extraction and pre-storage filtering.
  - `cleaner.py`: `TopicCleaner` — tokenization, stopwords, metadata rejection, malformed phrase checks.
  - `normalizer.py`: canonicalization rules and display mapping (aliases, display names, e.g., "js" -> "JavaScript", "html css" -> "HTML/CSS").
  - `graph_builder.py`: Aggregates per-item topic payloads into the semantic graph, computes related edges using embedding similarity hints.
  - `ontology_mapper.py`: Domain and known-topic lookups, mapping noisy text to canonical topics.
- `storage/`:
  - `json_store.py`: Stores Classroom items under `data/{materials,assignments,announcements}` as JSON.
  - `topics_store.py`: Per-item topic payload persistence under `data/topics/{type}/{course}_{item}.json`.
  - `topic_graph.py` / `graph_storage.py`: Persisted topic graph at `data/topic_graph.json`.
- `student/`:
  - `knowledge_store.py`: Stores `data/knowledge_state.json` with topic mastery/confidence and evidence.
  - `mastery_engine.py`, `knowledge_tracker.py`: Apply inference rules to update student mastery based on retrieved evidence and assessments.
- `rag/prompts.py`: Centralized prompt builders. Includes the new `topic_extraction_prompt()` for strict JSON extraction and several grounded QA templates.
- `tests/`: Unit tests and regression tests (topic graph pipeline, RAG prompts, CLI tests).

4. Data Flow (step-by-step)
---------------------------
1. Polling cycle in `main.py` fetches Classroom items and attachments, downloads attachments to `data/`.
2. For each new material/assignment, `TopicExtractor.extract()` builds a source text (title+description+attachment text). Heuristic candidate phrases are found using token and phrase heuristics.
3. Candidate phrases are filtered by `TopicCleaner` (metadata removed, length and noise filters applied).
4. Optionally (if enabled), an LLM-backed extractor is called with `topic_extraction_prompt()` which MUST return JSON with `main_topics`, `subtopics`, and `prerequisites`. The extractor strictly parses JSON and rejects metadata.
5. `TopicNormalizer` canonicalizes phrases and aliases (e.g., `js` -> `JavaScript`).
6. `SemanticTopicGraphBuilder` aggregates per-item payloads, counts evidence, builds related edges using embedding similarities, and saves `data/topic_graph.json`.
7. `KnowledgeStore` entries are updated from graph evidence and rules to produce `data/knowledge_state.json`.
8. `ChatAssistant` builds an in-memory embedding index (from summaries or item texts) and answers queries by retrieving top-k chunks and calling `LocalLLM.generate()` with grounded prompts.

5. Techniques and Rationale
--------------------------
- Hybrid extraction: simple heuristics catch many noun-phrase candidates (fast, deterministic). The LLM stage provides disambiguation and structure (primary, subtopics, prerequisites) but is constrained by a strict JSON prompt to reduce hallucination.
- Strong cleaning & canonicalization: removes filenames/titles/"Attachment Text" noise and normalizes synonyms/aliases to yield a concise graph.
- RAG grounding: answers are generated only from retrieved context; low-evidence queries are blocked by a confidence threshold and return an explicit refusal to answer (avoids hallucination).
- Embedding-based merging: semantically similar candidate phrases are merged using sentence-transformer embeddings to deduplicate semantically equivalent topics.
- Ollama integration: allows running `qwen2.5:7b` locally via an Ollama daemon. The code prefers the Python `ollama` client (if installed) and falls back to `ollama run <model> <prompt>`.
- Performance & stability measures:
  - Embeddings and topic graph lookups are cached in-memory for chat sessions.
  - `MAIN_SAFE_MODE` and planned `--once` mode enable safe startup and one-shot runs for demos.
  - Timeout protection on LLM calls and thread-based inference timeouts to avoid blocking.

6. Configuration and Environment
--------------------------------
Primary environment variables (set them in `.env` or export in shell):
- `EMBEDDING_MODEL_PATH` — path to sentence-transformer (default `./models/minilm`).
- `LLM_BACKEND` — `ollama` | `transformers` | `llama_cpp`.
- `LLM_MODEL_PATH` — Ollama model name (e.g., `qwen2.5:7b`) or filesystem path for local model files.
- `RAG_TOP_K`, `RAG_MAX_NEW_TOKENS`, `RAG_TEMPERATURE` — retrieval & generation tuning.
- `TOPIC_EXTRACT_LLM_ENABLED` — enable LLM-based topic extraction (1/0).
- `MAIN_SAFE_MODE` — when `1`, `python main.py` will run startup checks and exit 0.

7. How to Run (local demo)
--------------------------
1. Activate venv:

   ```powershell
   & .venv\Scripts\Activate.ps1
   ```
2. Ensure `.env` contains at least:

   ```dotenv
   EMBEDDING_MODEL_PATH=./models/minilm
   LLM_BACKEND=ollama
   LLM_MODEL_PATH=qwen2.5:7b
   ```
3. Quick startup check:

   ```powershell
   $env:MAIN_SAFE_MODE='1'
   python main.py
   ```

4. Run normal demo (starts polling + scheduler):

   ```powershell
   python main.py
   ```

5. Run the interactive chat CLI:

   ```powershell
   python -m chat.cli
   # or batch mode
   python -m chat.cli --batch-file tests/chat_cli_stress_questions.txt --batch-output tests/chat_cli_stress_results.json --show-timing
   ```

8. Demo Tips and Presentation Notes
----------------------------------
- Use `MAIN_SAFE_MODE=1` during slides to show startup checks quickly.
- For live QA, ensure the Ollama daemon is running and `qwen2.5:7b` is available (or set `LLM_BACKEND` to `transformers` with a local model).
- Emphasize the evidence list appended to answers — this demonstrates grounded reasoning and avoids hallucination.
- Use the `student_profile_summary()` method (available on `ChatAssistant`) to show known/weak/recent topics.

9. Limitations & Known Issues
----------------------------
- Topic extraction quality depends on the input text; short titles may be ambiguous. The strict cleaning rules reduce noise but may drop borderline concepts.
- Ollama CLI fallback used when the Python client is absent; the client provides better streaming semantics.
- The system uses local files for embeddings and models — performance will vary by hardware.

10. Files changed for demo-quality cleanup
-----------------------------------------
- `topic_graph/cleaner.py` — stronger metadata filtering & malformed phrase rejection
- `topic_graph/normalizer.py` — additional canonicalization (HTML/CSS alias)
- `rag/prompts.py` — strict `topic_extraction_prompt()` with JSON schema and few-shot examples
- `topic_graph/extractor.py` — now uses strict JSON extraction and validates schema
- `chat/assistant.py` — evidence scoring, low-confidence guard, formatted responses, `student_profile_summary()` helper
- `main.py` — `MAIN_SAFE_MODE` safe startup option

11. Next recommended improvements (optional)
-----------------------------------------
- Install the Python `ollama` client in the venv (pip install ollama) so the backend prefers the client.
- Add a `--once` CLI flag to `main.py` to run a single polling cycle and exit (good for deterministic demos).
- Add a small end-to-end integration test that runs `python main.py --once` against `backups/` dataset.
- Expand `topic_graph/normalizer.py` aliases to cover domain-specific synonyms for your curriculum.

Contact & Support
-----------------
If you'd like, I can:
- Add a `scripts/rebuild_topics.py` convenience script.
- Install the Python `ollama` client in the venv and verify client-mode LLM calls.
- Add `--once` and a demo README with step-by-step presentation notes.

---
Generated on 2026-05-29 by the development assistant.
