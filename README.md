# Hermes Classroom Agent

Lightweight agent to integrate Google Classroom with a local RAG-backed assistant, topic extraction, and a topic-graph-based recommendation pipeline.

This repository provides tools to fetch Classroom items (announcements, materials, coursework), extract topics from documents, build a topic graph, and answer classroom-related questions using intent-based routing and lightweight retrieval.

Features

- Google Classroom integration via `classroom/classroom_client.py`.
- Intent-based routing with embeddings in `chat/router.py`.
- Topic extraction (`rag/topic_extractor.py`) and graph building (`student/topic_graph_builder.py`).
- Local RAG pipeline using models in `models/` (optional) and JSON-backed storage in `data/` and `storage/`.

Quickstart

1. Create a virtual environment and activate it (Windows):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
pytest -q
```

4. Run the polling loop (requires `credentials.json` and Classroom API enabled):

```bash
python main.py
```

Documentation

- Project overview and architecture: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)
- Example scripts: see `scripts/backfill_submissions.py` for a backfill example.

Notes

- The repository can run in a reduced mode using the JSON files in `data/` for testing without external APIs or large local models.
- If you want to generate or edit Word materials locally, install `python-docx` (`pip install python-docx`). A helper created five test `.docx` files in `test_materials/` during an earlier session.

Contributing

Please open issues or PRs against the `main` branch. For changes that touch data formats (topics, payloads, stored JSON), include migration notes or unit tests.

License

Check the repository root for licensing information.
