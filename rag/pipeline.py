"""RAG pipeline for summarizing new materials offline."""

import logging
import time
from pathlib import Path
from typing import List

import config
from rag.attachments import AttachmentTextExtractor
from rag.embeddings import EmbeddingModel
from rag.index import EmbeddingIndex
from rag.llm import LocalLLM
from storage.json_store import JsonStore
from storage.summary_store import SummaryStore


class RagPipeline:
    """Coordinates retrieval and summarization for Classroom materials."""

    def __init__(self):
        self.enabled = config.RAG_ENABLED
        self.embedding_model = None
        self.embedding_index = None
        self.llm = None
        self.summary_store = SummaryStore(config.RAG_SUMMARIES_PATH)
        self.attachment_extractor = AttachmentTextExtractor(
            config.BASE_DIR, max_chars=config.PDF_EXTRACT_MAX_CHARS
        )

        if not self.enabled:
            logging.info("RAG pipeline disabled.")
            return

        if not config.EMBEDDING_MODEL_PATH or not config.LLM_MODEL_PATH:
            logging.warning("RAG model paths are not configured.")
            self.enabled = False
            return

        if not Path(config.EMBEDDING_MODEL_PATH).exists():
            logging.warning("Embedding model path not found: %s", config.EMBEDDING_MODEL_PATH)
            self.enabled = False
            return

        if not Path(config.LLM_MODEL_PATH).exists():
            logging.warning("LLM model path not found: %s", config.LLM_MODEL_PATH)
            self.enabled = False
            return

        try:
            self.embedding_model = EmbeddingModel(
                config.EMBEDDING_MODEL_PATH, device=config.RAG_DEVICE
            )
            self.embedding_index = EmbeddingIndex(
                config.RAG_INDEX_PATH, self.embedding_model
            )
            self.llm = LocalLLM(
                config.LLM_MODEL_PATH,
                device=config.RAG_DEVICE,
                max_new_tokens=config.RAG_MAX_NEW_TOKENS,
                temperature=config.RAG_TEMPERATURE,
            )
        except Exception:
            logging.exception("Failed to initialize RAG models.")
            self.enabled = False

    def process_new_materials(
        self, json_store: JsonStore, material_ids: List[str]
    ) -> int:
        if not self.enabled:
            return 0
        if not material_ids:
            return 0

        materials = json_store.get_items_by_ids("materials", material_ids)
        if not materials:
            return 0

        added = self.embedding_index.upsert_items(materials, self._build_text)
        if added:
            logging.info("RAG index updated with %s new materials.", added)

        summaries_created = 0
        for material in materials:
            if self.summary_store.has_summary(material):
                continue
            logging.info(
                "Summarizing material %s (%s)",
                material.get("title", ""),
                material.get("id", ""),
            )
            query = self._build_query(material)
            contexts = self.embedding_index.search(query, config.RAG_TOP_K)
            summary, formats = self._summarize_material(material, contexts)
            self.summary_store.upsert_summary(material, summary, contexts, formats)
            summaries_created += 1

        return summaries_created

    def _build_text(self, material: dict) -> str:
        attachments = material.get("attachment_paths", [])
        attachment_text = ", ".join(attachments)
        extracted_text = ""
        if config.PDF_EXTRACT_ENABLED:
            extracted_text, _ = self.attachment_extractor.extract(attachments)
        return (
            f"Course: {material.get('course_name', '')}\n"
            f"Title: {material.get('title', '')}\n"
            f"Description: {material.get('description', '')}\n"
            f"Attachments: {attachment_text}\n"
            f"Extracted Text: {extracted_text}"
        )

    def _build_query(self, material: dict) -> str:
        return f"{material.get('title', '')} {material.get('description', '')}".strip()

    def _build_prompt(self, material: dict, contexts: List[dict]) -> tuple:
        context_block = self._build_context_block(contexts)
        attachments = material.get("attachment_paths", [])
        extracted_text = ""
        formats = []
        if config.PDF_EXTRACT_ENABLED:
            extracted_text, formats = self.attachment_extractor.extract(attachments)
        prompt = self._build_prompt_text(
            material,
            formats,
            extracted_text,
            context_block,
            "You are a study assistant. Summarize the material in 3-5 bullets.\n"
            "Keep it concise and focus on actionable points.",
        )
        return prompt, formats

    def _summarize_material(self, material: dict, contexts: List[dict]) -> tuple[str, List[str]]:
        started_at = time.perf_counter()
        attachments = material.get("attachment_paths", [])
        formats: List[str] = []
        chunk_texts: List[str] = []
        if config.PDF_EXTRACT_ENABLED:
            chunk_texts, formats = self.attachment_extractor.extract_chunks(
                attachments,
                chunk_size=config.RAG_CHUNK_SIZE_CHARS,
                overlap=config.RAG_CHUNK_OVERLAP_CHARS,
                max_chars=config.RAG_EXTRACT_MAX_CHARS,
            )

        if not chunk_texts:
            logging.info(
                "No chunked attachment text for %s; summarizing from prompt context only.",
                material.get("title", ""),
            )
            prompt, formats = self._build_prompt(material, contexts)
            summary = self.llm.generate(prompt)
            logging.info(
                "Summarized %s in %.2fs using non-chunked prompt.",
                material.get("title", ""),
                time.perf_counter() - started_at,
            )
            return summary, formats

        total_chunks = len(chunk_texts)
        chunk_summaries: List[str] = []
        for index, chunk in enumerate(chunk_texts, start=1):
            chunk_started_at = time.perf_counter()
            logging.info(
                "Summarizing chunk %s/%s for %s (%s chars)",
                index,
                total_chunks,
                material.get("title", ""),
                len(chunk),
            )
            prompt = self._build_chunk_prompt(material, formats, chunk, index, total_chunks)
            summary = self.llm.generate(prompt)
            if summary:
                chunk_summaries.append(summary)
            logging.info(
                "Finished chunk %s/%s for %s in %.2fs",
                index,
                total_chunks,
                material.get("title", ""),
                time.perf_counter() - chunk_started_at,
            )

        if not chunk_summaries:
            prompt, formats = self._build_prompt(material, contexts)
            summary = self.llm.generate(prompt)
            logging.info(
                "Summarized %s in %.2fs using fallback prompt.",
                material.get("title", ""),
                time.perf_counter() - started_at,
            )
            return summary, formats

        final_summary = self._reduce_summaries(material, formats, contexts, chunk_summaries)
        logging.info(
            "Completed summarizing %s in %.2fs",
            material.get("title", ""),
            time.perf_counter() - started_at,
        )
        return final_summary, formats

    def _build_context_block(self, contexts: List[dict]) -> str:
        if not contexts:
            return ""
        context_lines = []
        for entry in contexts:
            context_lines.append(
                f"- {entry.get('course_name', '')}: {entry.get('text', '')}"
            )
        context_block = "\n".join(context_lines)
        max_chars = config.RAG_CONTEXT_MAX_CHARS
        if max_chars and len(context_block) > max_chars:
            context_block = context_block[:max_chars]
        return context_block

    def _build_prompt_text(
        self,
        material: dict,
        formats: List[str],
        extracted_text: str,
        context_block: str,
        instructions: str,
    ) -> str:
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
        ]
        if extracted_text:
            parts.append(f"Extracted Attachment Text:\n{extracted_text}")
        if context_block:
            parts.append("Retrieved Context:\n" + context_block)
        parts.append("Summary:\n")
        return "\n\n".join(parts)

    def _build_chunk_prompt(
        self,
        material: dict,
        formats: List[str],
        chunk_text: str,
        index: int,
        total: int,
    ) -> str:
        instructions = (
            "You are a study assistant. Summarize this excerpt in 3-5 bullets.\n"
            "Focus on key details without repeating boilerplate."
        )
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
            f"Excerpt {index}/{total}:\n{chunk_text}",
            "Summary:\n",
        ]
        return "\n\n".join(parts)

    def _build_reduce_prompt(
        self,
        material: dict,
        formats: List[str],
        summary_text: str,
        context_block: str,
    ) -> str:
        instructions = (
            "You are a study assistant. Combine the bullet summaries into 3-5 bullets.\n"
            "Keep it concise and remove duplicates."
        )
        parts = [
            instructions,
            f"Material Title: {material.get('title', '')}",
            f"Material Description: {material.get('description', '')}",
            f"Attachment Formats: {', '.join(formats) if formats else 'none'}",
            "Chunk Summaries:\n" + summary_text,
        ]
        if context_block:
            parts.append("Retrieved Context:\n" + context_block)
        parts.append("Summary:\n")
        return "\n\n".join(parts)

    def _reduce_summaries(
        self,
        material: dict,
        formats: List[str],
        contexts: List[dict],
        summaries: List[str],
    ) -> str:
        started_at = time.perf_counter()
        max_chars = config.RAG_COMBINE_MAX_CHARS
        current = [summary.strip() for summary in summaries if summary and summary.strip()]
        if not current:
            return ""

        while len(current) > 1 and max_chars and len(self._summaries_to_text(current)) > max_chars:
            batches = self._batch_summaries(current, max_chars)
            reduced = []
            logging.info(
                "Reducing %s chunk summaries into %s batch(es) for %s",
                len(current),
                len(batches),
                material.get("title", ""),
            )
            for batch in batches:
                prompt = self._build_reduce_prompt(
                    material,
                    formats,
                    self._summaries_to_text(batch),
                    "",
                )
                batch_started_at = time.perf_counter()
                reduced_summary = self.llm.generate(prompt)
                if reduced_summary:
                    reduced.append(reduced_summary)
                logging.info(
                    "Finished reduction batch for %s in %.2fs",
                    material.get("title", ""),
                    time.perf_counter() - batch_started_at,
                )
            current = reduced
            if not current:
                return ""

        context_block = self._build_context_block(contexts)
        final_prompt = self._build_reduce_prompt(
            material,
            formats,
            self._summaries_to_text(current),
            context_block,
        )
        final_summary = self.llm.generate(final_prompt)
        logging.info(
            "Final reduction for %s completed in %.2fs",
            material.get("title", ""),
            time.perf_counter() - started_at,
        )
        return final_summary

    def _summaries_to_text(self, summaries: List[str]) -> str:
        lines = []
        for summary in summaries:
            text = summary.strip()
            if not text:
                continue
            if text.startswith("-") or text.startswith("*"):
                lines.append(text)
            else:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def _batch_summaries(self, summaries: List[str], max_chars: int) -> List[List[str]]:
        batches: List[List[str]] = []
        current: List[str] = []
        current_len = 0
        for summary in summaries:
            entry = summary.strip()
            if not entry:
                continue
            entry_len = len(entry) + 3
            if current and current_len + entry_len > max_chars:
                batches.append(current)
                current = [entry]
                current_len = entry_len
            else:
                current.append(entry)
                current_len += entry_len
        if current:
            batches.append(current)
        return batches

if __name__ == "__main__":
    pipeline = RagPipeline()
    pipeline.process_new_materials(JsonStore(config.BASE_DIR), [])