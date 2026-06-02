import json
import re

class TopicExtractor:
    def __init__(self, llm):
        self.llm = llm

    def extract(self, item: dict, item_type: str) -> dict:
        prompt = self._build_prompt(item, item_type)
        response = self.llm.generate(prompt)
        return self._parse(response)

    def _parse(self, response: str) -> dict:
        clean = re.sub(r"```(?:json)?|```", "", response).strip()
        try:
            return json.loads(clean, strict=False)  # allows control characters in strings
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw response:\n{clean}")

    def _build_prompt(self, item, item_type):
        item = json.dumps(item, indent=2)
        return f"""
You are a topic extraction system.

Return ONLY valid JSON No markdown. No backticks. Summary should strictly one liner.

Item Type: {item_type}
Item:
{item}
Generate topics, skills, difficulty(begginer, intermediate, advanced based on your knowledge), summary and put it in the given format extract topics, skills, difficulty, summary, item id, item_type(given in prompt) and course id if available. if not then put unknown.

Output format:
{{
  "topics": [],
  "skills": [],
  "difficulty": "",
  "summary": "",
  "item_id": "",
  "course_id": "",
  "item_type": ""
}}
"""