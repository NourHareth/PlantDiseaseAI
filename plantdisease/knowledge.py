"""Disease knowledge base: human-readable information for each model class.

The content itself lives in ``data/disease_info.json``; this module only loads
and looks it up. Model labels are matched by a normalized key, so
``"Tomato___Spider_mites Two-spotted_spider_mite"`` and
``"Spider mites Two-spotted spider mite"`` resolve to the same entry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from plantdisease import config

_PREFIX_RE = re.compile(r"^tomato_{2,}", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

CATEGORIES = ("disease", "pest", "healthy", "unknown")


def normalize_label(label: str) -> str:
    """Return a stable lookup key for a class label.

    Strips the PlantVillage ``Tomato___`` prefix, lower-cases, and collapses
    every run of punctuation/underscores/spaces into a single space.
    """
    text = _PREFIX_RE.sub("", label.strip())
    text = _NON_ALNUM_RE.sub(" ", text.lower())
    return text.strip()


def fallback_display_name(label: str) -> str:
    """Readable name for a label that has no knowledge-base entry."""
    key = normalize_label(label)
    return key[:1].upper() + key[1:] if key else label


@dataclass(frozen=True)
class DiseaseInfo:
    key: str
    display_name: str
    category: str
    cause: str
    description: str
    symptoms: tuple[str, ...]
    actions: tuple[str, ...]

    @property
    def is_healthy(self) -> bool:
        return self.category == "healthy"


def _unknown(label: str) -> DiseaseInfo:
    return DiseaseInfo(
        key=normalize_label(label),
        display_name=fallback_display_name(label),
        category="unknown",
        cause="Not documented",
        description="No additional information is available for this class yet.",
        symptoms=(
            "Review the original image carefully",
            "The prediction may not reflect every visible symptom",
        ),
        actions=(
            "Try a clearer image of a single leaf",
            "Seek expert advice if symptoms persist",
        ),
    )


class KnowledgeBase:
    """Lookup table from model class label to :class:`DiseaseInfo`."""

    def __init__(self, entries: dict[str, DiseaseInfo]):
        self._entries = entries

    @classmethod
    def from_json(cls, path: Path = config.DISEASE_INFO_PATH) -> "KnowledgeBase":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        entries: dict[str, DiseaseInfo] = {}
        for label, item in raw["classes"].items():
            key = normalize_label(label)
            category = item["category"]
            if category not in CATEGORIES:
                raise ValueError(f"{path}: unknown category {category!r} for {label!r}")
            if key in entries:
                raise ValueError(f"{path}: duplicate entry for {label!r}")
            entries[key] = DiseaseInfo(
                key=key,
                display_name=item["display_name"],
                category=category,
                cause=item["cause"],
                description=item["description"],
                symptoms=tuple(item["symptoms"]),
                actions=tuple(item["actions"]),
            )
        return cls(entries)

    def get(self, label: str) -> DiseaseInfo:
        """Information for ``label``; a generic placeholder if it is not documented."""
        return self._entries.get(normalize_label(label)) or _unknown(label)

    def display_name(self, label: str) -> str:
        return self.get(label).display_name

    def missing(self, labels: list[str]) -> list[str]:
        """Labels that have no entry (used by tests and start-up checks)."""
        return [label for label in labels if normalize_label(label) not in self._entries]

    def __len__(self) -> int:
        return len(self._entries)
