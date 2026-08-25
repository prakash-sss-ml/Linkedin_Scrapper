from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\u200b", " ").replace("\ufeff", " ")
    return WHITESPACE_RE.sub(" ", value).strip()


def normalized_key(value: str) -> str:
    value = clean_text(value).casefold()
    value = re.sub(r"[^a-z0-9@.+#/-]+", " ", value)
    return WHITESPACE_RE.sub(" ", value).strip()


def is_fuzzy_duplicate(candidate: str, existing: list[str], threshold: float = 0.92) -> bool:
    candidate_key = normalized_key(candidate)
    if not candidate_key:
        return True

    for previous in existing[-25:]:
        previous_key = normalized_key(previous)
        if candidate_key == previous_key:
            return True
        if min(len(candidate_key), len(previous_key)) < 12:
            continue
        if SequenceMatcher(None, candidate_key, previous_key).ratio() >= threshold:
            return True
    return False


def split_bullet_line(value: str) -> list[str]:
    value = clean_text(value)
    if not value:
        return []
    parts = re.split(r"\s*(?:·|•|\||—)\s*", value)
    return [clean_text(part) for part in parts if clean_text(part)]
