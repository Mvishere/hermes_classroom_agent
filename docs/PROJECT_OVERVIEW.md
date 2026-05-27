Hermes Classroom Agent — Project Overview

This document explains the purpose, architecture, and key workflows of the Hermes Classroom Agent repository so contributors and reviewers can quickly understand how the system functions.

Purpose
- Assist with Google Classroom operations (announcements, materials, assignments) and provide an embeddable RAG-backed chat assistant for classroom-related queries.
- Extract topics from materials and build a topic graph to drive recommendations and knowledge tracking.

High-level Architecture
- auth/ — Google OAuth helpers. `google_auth.py` wraps credentials and token refresh.
- classroom/ — API wrapper and domain helpers. Key file: `classroom_client.py` (Classroom API calls), `coursework.py` (assignment storage/logic), `announcements.py`, `materials.py`.
- chat/ — Conversation layer and intent router:
  - `router.py` — Embedding-based intent classifier that routes queries to tools.
  - `tools/` — Small, focused responders:
    - `factual_tool.py` — Direct answers from stored JSON facts.
    - `summarization_tool.py` — Returns stored summaries.
    - `general_tool.py` — RAG/LLM fallback.
  - `assistant.py` — Core RAG answer engine used by the general tool.
- rag/ — Retrieval + generation helpers:
  - `embeddings.py` — Embedding model wrapper.
  - `llm.py` — Local LLM interface.
  - `topic_extractor.py` — Extracts topic payloads from material text.
  - `index.py`, `pipeline.py` — Retrieval and pipeline orchestration.
- student/ — Topic graph and student-facing tooling:
  - `topic_graph_builder.py` — Builds topic relations from item payloads.
  - `knowledge_tracker.py` (if present) — Tracks mastery signals.
- storage/ — JSON-backed persistence helpers:
  - `file_storage.py`, `json_store.py`, `summary_store.py`, `topics_store.py`, `user_status.py`.
- scripts/ — One-off scripts such as `backfill_submissions.py` for submission state fixes.
- `main.py` — Polling loop that orchestrates topic extraction, topic-graph updates, and Classroom refreshes.

Data layout
- `data/` holds the canonical JSON used for tests and running the system: `assignments/`, `announcements/`, `materials/`, `rag/`, `state/`.

Core Workflows

- Polling cycle (`main.py`):
  1. Fetch Classroom items (materials, announcements, assignments).
  2. Extract topics for new/updated items via `rag/topic_extractor.py`.
  3. Persist topic payloads to `storage/topics_store.py`.
  4. Rebuild topic graph with `student/topic_graph_builder.py`.
  5. Update `storage/user_status` and recommendations as appropriate.

- Intent-based chat routing:
  - Incoming queries are embedded using the embedding model.
  - `chat/router.py` classifies the intent against pre-built intent vectors.
  - The router dispatches to `factual_tool`, `summarization_tool`, or the `general_tool` (RAG + LLM) as appropriate.

- Topic extraction & graph building:
  - `topic_extractor.py` applies keyword extraction (stopwords and heuristics) and can optionally use the LLM for JSON extraction.
  - Co-occurrence and mastery deltas are used by `topic_graph_builder.py` to create edges and weights.

Generating test materials
- A small helper was used to produce five `.docx` test files in `test_materials/` to validate topic extraction and graph overlaps. Upload those documents to a Classroom course and then run the polling cycle to observe topic payloads in `data/rag/` and the produced `data/topic_graph.json`.

Running locally (quickstart)
1. Create and activate virtual environment (Windows example):

```
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run tests:

```
pytest -q
```

4. Run the polling loop (requires configured `credentials.json` and Classroom API access):

```
python main.py
```

Notes: some features expect local models under `models/` (embeddings and LLM); other flows fall back to small test data in `data/` when models or keys are absent.

Key files to inspect when troubleshooting
- `main.py` — orchestration and polling.
- `chat/router.py` — intent routing logic and examples.
- `rag/topic_extractor.py` — how topics are derived from materials.
- `student/topic_graph_builder.py` — how the graph is constructed from payloads.
- `classroom/classroom_client.py` — Classroom API wrapper used across scripts.

Troubleshooting & common tasks
- Missing `python-docx` stopped `.docx` generation earlier; it can be installed via `pip install python-docx`.
- If topic extraction is noisy, check `TOPIC_EXTRACT_*` settings in `config.py` and the stopword list used by `topic_extractor.py`.

Next steps and contributions
- Improve topic extraction precision (add more examples, tune stopwords, enable LLM-assisted JSON extraction selectively).
- Add unit tests for `topic_extractor` and `topic_graph_builder` to prevent regression.

---
This file is a concise starting point. If you want, I can expand sections with file-level references, sequence diagrams, or an onboarding checklist.
