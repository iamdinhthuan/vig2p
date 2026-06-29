from __future__ import annotations

from .g2p import G2P
from .normalizer import Normalizer


class Pipeline:
    def __init__(self) -> None:
        self.normalizer = Normalizer()
        self.g2p = G2P()

    def run(self, text: str | list[str], punc_norm: bool = False) -> str | list[str]:
        if not text:
            return "" if isinstance(text, str) else []
        normalized = self.normalizer.normalize(text, punc_norm=punc_norm)
        return self.g2p.convert(normalized)
