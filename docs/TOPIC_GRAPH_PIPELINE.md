# Semantic Topic Graph Pipeline

This repository now builds topic graphs from educational concepts instead of raw keyword co-occurrence.

## Why co-occurrence graphs failed

Raw co-occurrence treats every repeated token as a potential topic. That produces noisy nodes such as `what`, `when`, `test`, and `page`, and it creates dense edges between unrelated items that merely appear in the same document. The result is a flat graph with weak prerequisite signals and poor recommendations.

## Why semantic educational graphs are better

The new pipeline extracts noun-phrase candidates, filters low-information terms, canonicalizes duplicates, maps concepts to a curated ontology, and keeps only weighted relationships with enough semantic confidence. That means the graph can represent real educational structure such as `HTML -> CSS -> Responsive Design -> Media Queries` instead of only counting words that happen to repeat.

## What the pipeline produces

Each item is structured into:

- `primary_topic`
- `subtopics`
- `prerequisites`
- `domain`
- `difficulty`
- weighted `related_topics`

The graph storage format is hierarchical and forward-compatible:

```json
{
  "version": 2,
  "generated_at": "2026-05-28T00:00:00Z",
  "topics": {
    "Responsive Design": {
      "canonical_name": "responsive design",
      "display_name": "Responsive Design",
      "domain": "Web Development",
      "difficulty": "medium",
      "parent_topic": "CSS",
      "prerequisites": ["CSS Basics"],
      "subtopics": ["Media Queries", "Flexbox"],
      "related_topics": [
        {"topic": "CSS Grid", "weight": 0.83}
      ]
    }
  }
}
```

## Local-only components

The pipeline stays local-first:

- keyword cleaning and normalization are rule-based
- ontology mapping is curated in-repo
- semantic similarity uses the local embedding model already configured in `.env`
- optional LLM structuring uses the local causal model path from `.env`

## Compatibility notes

Existing callers can still use the `TopicGraph`, `TopicGraphBuilder`, and `TopicExtractor` entry points. Those are now compatibility wrappers over the semantic implementation in `topic_graph/`.
