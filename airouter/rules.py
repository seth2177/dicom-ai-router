"""Routing rules: decide which AI model (if any) a study goes to, from its DICOM header."""
from __future__ import annotations

import re

from pydicom import Dataset

from .config import Rule


def match_rule(ds: Dataset, rules: list[Rule]) -> Rule | None:
    """Return the first rule that matches, else None.

    `match`: every `DicomKeyword: regex` must match (AND).
    `match_any`: at least one must match (OR) -- used for QA studies, whose
    "this is a phantom" signal can be in any of several free-text fields.
    Regexes are searched case-insensitively.
    A missing tag never matches -- real headers are often incomplete
    (BodyPartExamined is blank on many scanners), which is why the
    sample config has a second rule that falls back to StudyDescription.
    """
    for rule in rules:
        if not all(_cond(ds, kw, p) for kw, p in rule.match.items()):
            continue
        if rule.match_any and not any(_cond(ds, kw, p) for kw, p in rule.match_any.items()):
            continue
        return rule
    return None


def _cond(ds: Dataset, keyword: str, pattern: str) -> bool:
    value = ds.get(keyword)
    if value is None or str(value).strip() == "":
        return False
    return re.search(pattern, str(value), re.IGNORECASE) is not None
