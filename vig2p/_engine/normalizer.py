from __future__ import annotations

from .punc import apply_punc_norm


class Normalizer:
    def normalize(self, text: str | list[str], punc_norm: bool = False) -> str | list[str]:
        if isinstance(text, list):
            return [self.normalize(item, punc_norm=punc_norm) for item in text]
        return apply_punc_norm(text) if punc_norm else text
