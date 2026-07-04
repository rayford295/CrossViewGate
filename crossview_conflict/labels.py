from __future__ import annotations

import re
from typing import Optional


CANONICAL_CATEGORIES = (
    "No Damage",
    "Affected (1-9%)",
    "Minor (10-25%)",
    "Major (26-50%)",
    "Destroyed (>50%)",
    "Inaccessible",
)

IAN_HURRICANE_CATEGORIES = (
    "0_MinorDamage",
    "1_ModerateDamage",
    "2_SevereDamage",
)

MILTON_HURRICANE_CATEGORIES = (
    "mild_damage",
    "moderate_damage",
    "severe_damage",
)

_CATEGORY_ALIASES = {
    "no_damage": "No Damage",
    "affected_1_9": "Affected (1-9%)",
    "affected": "Affected (1-9%)",
    "minor_10_25": "Minor (10-25%)",
    "minor": "Minor (10-25%)",
    "major_26_50": "Major (26-50%)",
    "major": "Major (26-50%)",
    "destroyed_50": "Destroyed (>50%)",
    "destroyed": "Destroyed (>50%)",
    "inaccessible": "Inaccessible",
}

_BINARY_SCHEMES = {
    "operational": {
        "No Damage": 0,
        "Affected (1-9%)": 0,
        "Minor (10-25%)": 1,
        "Major (26-50%)": 1,
        "Destroyed (>50%)": 1,
        "Inaccessible": None,
    },
    "sensitive": {
        "No Damage": 0,
        "Affected (1-9%)": 1,
        "Minor (10-25%)": 1,
        "Major (26-50%)": 1,
        "Destroyed (>50%)": 1,
        "Inaccessible": None,
    },
}


def _label_key(raw_category: object) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(raw_category).strip().lower())).strip("_")


def normalize_category(raw_category: object) -> str:
    if raw_category is None:
        raise ValueError("Category cannot be None.")
    normalized = _label_key(raw_category)
    if normalized in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[normalized]
    for canonical in CANONICAL_CATEGORIES:
        if normalized == _label_key(canonical):
            return canonical
    raise KeyError(f"Unknown damage category: {raw_category!r}")


def to_binary_label(raw_category: object, scheme: str = "operational") -> Optional[int]:
    canonical = normalize_category(raw_category)
    if scheme not in _BINARY_SCHEMES:
        raise KeyError(f"Unknown binary scheme: {scheme}")
    return _BINARY_SCHEMES[scheme][canonical]


def to_binary_name(raw_category: object, scheme: str = "operational") -> str:
    binary_label = to_binary_label(raw_category, scheme=scheme)
    if binary_label is None:
        return "ignored"
    return "damage" if binary_label == 1 else "no_damage"


def to_multiclass_label(raw_category: object) -> int:
    canonical = normalize_category(raw_category)
    return CANONICAL_CATEGORIES.index(canonical)


def to_multiclass_name(raw_category: object) -> str:
    return normalize_category(raw_category)


def build_class_mapping(class_names: tuple[str, ...] | list[str]) -> dict[str, int]:
    return {str(name): index for index, name in enumerate(class_names)}
