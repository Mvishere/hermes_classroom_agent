from __future__ import annotations

import logging
from typing import List

import config

from rag.attachments import AttachmentTextExtractor
from rag.embeddings import EmbeddingModel
from rag.llm import LLM

from storage.json_store import JsonStore
from storage.summary_store import SummaryStore
from vector_store.chroma_store import ChromaStore



class RagPipeline:
    """
    RAG Pipeline using ChromaDB + Ollama embeddings.
    """

    def __init__(self):
        self.enabled = config.RAG_ENABLED

        self.summary_store = SummaryStore(config.RAG_SUMMARIES_PATH)

        self.attachment_extractor = AttachmentTextExtractor(
            config.BASE_DIR,
            max_chars=config.PDF_EXTRACT_MAX_CHARS
        )

        self.embedding_model = None
        self.chroma_store = None
        self.llm = None

        if not self.enabled:
            logging.info("RAG disabled.")
            return

        # ---------------- EMBEDDINGS (OLLAMA) ----------------
        self.embedding_model = EmbeddingModel(
            model_name="mxbai-embed-large"
        )

        # ---------------- CHROMA DB ----------------
        self.chroma_store = ChromaStore(
            persist_dir=config.CHROMA_PERSIST_DIR,
        )

        # ---------------- LLM ----------------
        self.llm = LLM(
            config.LLM_MODEL_PATH,
            device=config.RAG_DEVICE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
        )

    # =========================================================
    # SIMPLE CHUNKING (TEMPORARY BUT IMPORTANT)
    # =========================================================
    def _chunk_text(self, text: str, chunk_size: int = 800) -> List[str]:
        return [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
        ]

    # =========================================================
    # INGESTION → CHROMA
    # =========================================================
    def process_new_materials(
        self,
        json_store: JsonStore,
        material_ids: List[str]
    ) -> int:

        if not self.enabled or not material_ids:
            return 0

        materials = json_store.get_items_by_ids("materials", material_ids)
        if not materials:
            return 0

        chunks_to_store = []
        embeddings_to_store = []

        # ---------------- INGEST + CHUNK + EMBED ----------------
        for material in materials:

            raw_text = self._build_text(material)
            chunks = self._chunk_text(raw_text)

            if not chunks:
                continue

            # 🔥 batch embed (much faster than per-chunk calls if supported)
            embeddings = self.embedding_model.encode(chunks)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):

                chunks_to_store.append({
                    "id": f"{material['id']}_{i}",
                    "text": chunk,
                    "course_id": material.get("course_id", ""),
                    "item_id": material.get("id", ""),
                    "item_type": "material"
                })

                embeddings_to_store.append(emb)

        # ---------------- STORE IN CHROMA ----------------
        if chunks_to_store:
            self.chroma_store.upsert_chunks(
                chunks_to_store,
                embeddings_to_store
            )

        logging.info(
            "Chroma updated: %s materials -> %s chunks",
            len(materials),
            len(chunks_to_store)
        )

        # ---------------- SUMMARIZATION ----------------
        summaries_created = 0

        for material in materials:

            if self.summary_store.has_summary(material):
                continue

            query = self._build_query(material)
            query_embedding = self.embedding_model.encode([query])[0]

            contexts = self.chroma_store.search(
                query_embedding=query_embedding,
                top_k=config.RAG_TOP_K
            )

            summary, formats = self._summarize_material(material, contexts)

            self.summary_store.upsert_summary(
                material,
                summary,
                contexts,
                formats
            )

            summaries_created += 1

        return summaries_created
    # =========================================================
    # TEXT BUILDING
    # =========================================================
    def _build_text(self, material: dict) -> str:
        attachments = material.get("attachment_paths", [])
        extracted_text = ""

        if config.PDF_EXTRACT_ENABLED:
            extracted_text, _ = self.attachment_extractor.extract(attachments)

        return (
            f"Course: {material.get('course_name', '')}\n"
            f"Title: {material.get('title', '')}\n"
            f"Description: {material.get('description', '')}\n"
            f"Extracted Text: {extracted_text}"
        )

    def _build_query(self, material: dict) -> str:
        return f"{material.get('title', '')} {material.get('description', '')}"

    # =========================================================
    # SUMMARIZATION
    # =========================================================
    def _summarize_material(self, material: dict, contexts: List[dict]):
        context_text = "\n\n".join(c.get("text", "") for c in contexts)

        prompt = f"""
You are a study assistant.

Material:
{material.get('title','')}

Context:
{context_text}

Summarize in 3-5 bullet points.
"""

        summary = self.llm.generate(prompt)
        return summary, []