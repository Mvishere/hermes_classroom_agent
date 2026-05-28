"""Curated educational ontology and mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from topic_graph.normalizer import TopicNormalizer


@dataclass(frozen=True)
class OntologyEntry:
    """Represents a canonical educational concept."""

    canonical: str
    display_name: str
    domain: str
    parent: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)
    subtopics: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


_ONTOLOGY: Dict[str, List[OntologyEntry]] = {
    "web_development": [
        OntologyEntry("html", "HTML", "Web Development", subtopics=["html semantics", "forms", "tables"], aliases=["hypertext markup language"]),
        OntologyEntry("css", "CSS", "Web Development", prerequisites=["HTML"], subtopics=["flexbox", "css grid", "responsive design", "media queries"], aliases=["cascading style sheets"]),
        OntologyEntry("flexbox", "Flexbox", "Web Development", parent="CSS", prerequisites=["CSS"], aliases=["flex box", "flex layout"]),
        OntologyEntry("css grid", "CSS Grid", "Web Development", parent="CSS", prerequisites=["CSS"], aliases=["grid layout"]),
        OntologyEntry("responsive design", "Responsive Design", "Web Development", parent="CSS", prerequisites=["CSS", "Media Queries"], aliases=["responsive web design"]),
        OntologyEntry("media queries", "Media Queries", "Web Development", parent="CSS", prerequisites=["CSS"], aliases=["media query"]),
        OntologyEntry("javascript", "JavaScript", "Web Development", prerequisites=["HTML", "CSS"], subtopics=["functions", "events", "dom manipulation", "conditionals"], aliases=["js"]),
        OntologyEntry("dom", "DOM", "Web Development", parent="JavaScript", prerequisites=["HTML", "JavaScript"], aliases=["document object model"]),
        OntologyEntry("dom manipulation", "DOM Manipulation", "Web Development", parent="JavaScript", prerequisites=["DOM", "JavaScript"], aliases=["dom" ]),
        OntologyEntry("event", "Event", "Web Development", parent="JavaScript", prerequisites=["JavaScript"], aliases=["events"]),
        OntologyEntry("javascript events", "JavaScript Events", "Web Development", parent="JavaScript", prerequisites=["JavaScript", "DOM"], aliases=["event handling"]),
        OntologyEntry("function", "Function", "Web Development", parent="JavaScript", prerequisites=["JavaScript"], aliases=["functions"]),
        OntologyEntry("query", "Query", "Web Development", parent="JavaScript", prerequisites=["JavaScript"], aliases=["queries"]),
    ],
    "computer_science": [
        OntologyEntry("algorithm", "Algorithm", "Computer Science", subtopics=["searching", "sorting", "complexity"]),
        OntologyEntry("data structure", "Data Structure", "Computer Science", subtopics=["arrays", "lists", "trees", "graphs"]),
    ],
}


class OntologyMapper:
    """Maps extracted phrases to curated educational concepts."""

    def __init__(self, normalizer: Optional[TopicNormalizer] = None) -> None:
        self.normalizer = normalizer or TopicNormalizer()
        self._entries: Dict[str, OntologyEntry] = {}
        self._alias_index: Dict[str, str] = {}
        for domain_entries in _ONTOLOGY.values():
            for entry in domain_entries:
                self._entries[entry.canonical] = entry
                self._alias_index[self.normalizer.canonicalize(entry.display_name)] = entry.canonical
                for alias in entry.aliases:
                    self._alias_index[self.normalizer.canonicalize(alias)] = entry.canonical

    def known_topics(self) -> List[str]:
        return sorted({entry.display_name for entry in self._entries.values()})

    def entry_for(self, topic: str) -> Optional[OntologyEntry]:
        canonical = self.normalizer.canonicalize(topic)
        if not canonical:
            return None
        mapped = self._alias_index.get(canonical, canonical)
        return self._entries.get(mapped)

    def map_topic(self, topic: str) -> str:
        entry = self.entry_for(topic)
        if entry:
            return entry.display_name
        return self.normalizer.display_name(topic)

    def domain_for(self, topic: str) -> str:
        entry = self.entry_for(topic)
        return entry.domain if entry else "General"

    def parent_for(self, topic: str) -> str:
        entry = self.entry_for(topic)
        if not entry:
            return ""
        if entry.parent:
            return self.map_topic(entry.parent)
        if entry.domain:
            return entry.domain
        return ""

    def prerequisites_for(self, topic: str) -> List[str]:
        entry = self.entry_for(topic)
        if not entry:
            return []
        return self.normalizer.merge(entry.prerequisites)

    def subtopics_for(self, topic: str) -> List[str]:
        entry = self.entry_for(topic)
        if not entry:
            return []
        return self.normalizer.merge(entry.subtopics)

    def is_known(self, topic: str) -> bool:
        return self.entry_for(topic) is not None

    def all_entries(self) -> List[OntologyEntry]:
        return list(self._entries.values())
