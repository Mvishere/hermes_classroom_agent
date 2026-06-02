import json
import logging
from pathlib import Path

import pandas as pd

from rag.pipeline import RagPipeline


# -----------------------------
# DATA LOADERS
# -----------------------------

def load_all_topics():
    """
    Loads all JSON topic files from:
    data/topics/assignments/
    data/topics/materials/
    """
    base_path = Path("data") / "topics"

    if not base_path.exists():
        return {}

    result = {
        "dict": []
    }

    for subfolder in base_path.iterdir():
        if not subfolder.is_dir():
            continue

        category = subfolder.name

        for file in subfolder.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    result[category].extend(data)
                else:
                    result[category].append(data)

            except Exception as e:
                logging.exception(f"Failed loading topic file {file}: {e}")

    return result


def load_csv_context():
    """
    Loads CSV metadata:
    data/announcements.csv
    data/courses.csv
    data/materials.csv
    """
    base = Path("data")

    files = {
        "announcements": base / "announcements.csv",
        "courses": base / "courses.csv",
        "materials": base / "materials.csv",
    }

    data = {}

    for key, path in files.items():
        if path.exists():
            try:
                df = pd.read_csv(path)
                data[key] = df.to_dict(orient="records")
            except Exception as e:
                data[key] = [{"error": str(e)}]
        else:
            data[key] = []

    return data


# -----------------------------
# CHAT CLI
# -----------------------------

class RagChatCLI:
    def __init__(self):
        self.rag = RagPipeline()
        self.llm = self.rag.llm

    def ask(self, query: str) -> str:
        # 1. Embed query
        query_embedding = self.rag.embedding_model.encode([query])[0]

        # 2. Retrieve from Chroma
        contexts = self.rag.chroma_store.search(
            query_embedding=query_embedding,
            top_k=5
        )

        context_text = "\n\n".join(
            c.get("text", "") for c in contexts
        )

        # 3. Load structured data
        topics_data = load_all_topics()
        csv_data = load_csv_context()

        # 4. Build STUDY ASSISTANT PROMPT
        prompt = f"""
You are a STUDY ASSISTANT AI for a classroom learning system.

You must:
- Use ONLY the provided context
- Prefer structured reasoning
- If information is missing, clearly say so
- Keep answers simple and student-friendly
- Metadata is provided for reference but not all fields may be relevant

---

📚 TOPICS DATA:
{json.dumps(topics_data, indent=2)}

📊 COURSES (CSV):
{json.dumps(csv_data.get("courses", []), indent=2)}

📢 ANNOUNCEMENTS (CSV):
{json.dumps(csv_data.get("announcements", []), indent=2)}

📘 MATERIALS METADATA (CSV):
{json.dumps(csv_data.get("materials", []), indent=2)}

---

📖 RETRIEVED CONTEXT (RAG):
{context_text}

---

❓ QUESTION:
{query}

---

✍️ FINAL ANSWER:
"""

        return self.llm.generate(prompt)

    def run(self):
        print("\n🤖 Study Assistant Ready (type 'exit' to quit)\n")

        while True:
            query = input("You: ")

            if query.lower() in ["exit", "quit"]:
                break

            try:
                answer = self.ask(query)
                print(f"\nAgent: {answer}\n")
            except Exception as e:
                logging.exception("Chat failed")
                print(f"\nError: {e}\n")


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    RagChatCLI().run()