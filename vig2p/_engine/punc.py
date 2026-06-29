from __future__ import annotations


SHORT_SENTENCE_MAX_WORDS = 4
TRAILING_PUNCT = {",", ".", "!", "?", ";", ":", "\u2026", "\u2025", "\u2024"}
SENTENCE_END = {",", ".", "!", "?"}


def _word_count(text: str) -> int:
    return sum(1 for word in text.split() if any(char.isalnum() for char in word))


def apply_punc_norm(text: str) -> str:
    trimmed = text.rstrip()
    if not trimmed:
        return trimmed

    if _word_count(trimmed) <= SHORT_SENTENCE_MAX_WORDS:
        stripped = trimmed.rstrip("".join(TRAILING_PUNCT) + " \t\r\n")
        return "." if not stripped else f"{stripped}."

    return trimmed if trimmed[-1] in SENTENCE_END else f"{trimmed}."
