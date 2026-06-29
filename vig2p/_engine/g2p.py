from __future__ import annotations

import mmap
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .punc import apply_punc_norm


RE_TOKEN = re.compile(r"(?i)(<en>.*?</en>)|(\w+(?:['’]\w+)*)|([^\w\s])", re.UNICODE)
RE_TAG_CONTENT = re.compile(r"(\w+(?:['’]\w+)*)|([^\w\s])", re.UNICODE)
RE_TAG_STRIP = re.compile(r"(?i)</?en>", re.UNICODE)

VI_ACCENTS = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
VOWELS = "aeiouy" + VI_ACCENTS.replace("đ", "")
STOP_PUNCT = set(".!?;:()[]{}")


def _resource_path() -> Path:
    return Path(__file__).with_name("lexicon.bin")


def _has_vowel_and_consonant(text: str) -> bool:
    has_vowel = False
    has_consonant = False
    for char in text:
        lower = char.lower()
        if lower in VOWELS:
            has_vowel = True
        elif lower.isalpha():
            has_consonant = True
        if has_vowel and has_consonant:
            return True
    return False


class Lexicon:
    def __init__(self, path: str | Path) -> None:
        with Path(path).open("rb") as lexicon_file:
            self._mmap = mmap.mmap(lexicon_file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self._mmap) < 32 or self._mmap[0:4] != b"SEAP":
            raise ValueError("Invalid lexicon format")

        self.string_count = self._u32(8)
        self.merged_count = self._u32(12)
        self.common_count = self._u32(16)
        self.string_offsets_pos = self._u32(20)
        self.merged_pos = self._u32(24)
        self.common_pos = self._u32(28)
        self._strings: dict[int, str] = {}

    def _u32(self, offset: int) -> int:
        return struct.unpack_from("<I", self._mmap, offset)[0]

    def get_string(self, string_id: int) -> str:
        if string_id >= self.string_count:
            return ""
        if string_id in self._strings:
            return self._strings[string_id]

        off_ptr = self.string_offsets_pos + string_id * 4
        start = 32 + self._u32(off_ptr)
        end = self._mmap.find(b"\0", start)
        if end == -1:
            end = len(self._mmap)
        value = self._mmap[start:end].decode("utf-8", errors="strict")
        self._strings[string_id] = value
        return value

    def lookup_merged(self, word: str) -> str | None:
        low = 0
        high = self.merged_count - 1
        while low <= high:
            mid = (low + high) // 2
            ptr = self.merged_pos + mid * 8
            current_word = self.get_string(self._u32(ptr))
            if current_word == word:
                return self.get_string(self._u32(ptr + 4))
            if current_word < word:
                low = mid + 1
            else:
                high = mid - 1
        return None

    def lookup_common(self, word: str) -> tuple[str, str] | None:
        low = 0
        high = self.common_count - 1
        while low <= high:
            mid = (low + high) // 2
            ptr = self.common_pos + mid * 12
            current_word = self.get_string(self._u32(ptr))
            if current_word == word:
                return self.get_string(self._u32(ptr + 4)), self.get_string(self._u32(ptr + 8))
            if current_word < word:
                low = mid + 1
            else:
                high = mid - 1
        return None


@dataclass
class Token:
    lang: str
    content: str
    phone: str | None
    is_explicit_en: bool


class Engine:
    def __init__(self, lexicon_path: str | Path) -> None:
        self.lexicon = Lexicon(lexicon_path)
        self._merged_cache: dict[str, str] = {}
        self._common_cache: dict[str, tuple[str, str]] = {}
        self._missing_merged: set[str] = set()
        self._missing_common: set[str] = set()
        self._segmentation_cache: dict[tuple[str, str], str | None] = {}

    def cached_lookup_merged(self, word: str) -> str | None:
        if word in self._merged_cache:
            return self._merged_cache[word]
        if word in self._missing_merged:
            return None

        value = self.lexicon.lookup_merged(word)
        if value is not None:
            if len(self._merged_cache) >= 10_000:
                self._merged_cache.clear()
            self._merged_cache[word] = value
            return value
        if len(self._missing_merged) < 50_000:
            self._missing_merged.add(word)
        return None

    def cached_lookup_common(self, word: str) -> tuple[str, str] | None:
        if word in self._common_cache:
            return self._common_cache[word]
        if word in self._missing_common:
            return None

        value = self.lexicon.lookup_common(word)
        if value is not None:
            if len(self._common_cache) >= 5_000:
                self._common_cache.clear()
            self._common_cache[word] = value
            return value
        if len(self._missing_common) < 50_000:
            self._missing_common.add(word)
        return None

    def resolve_segment_phone(self, segment: str, lang: str) -> str | None:
        lower = segment.lower()
        merged = self.cached_lookup_merged(lower)
        if merged is not None:
            return merged.replace("<en>", "").strip()

        common = self.cached_lookup_common(lower)
        if common is not None:
            vi, en = common
            if lang == "en" and en:
                return en.replace("<en>", "").strip()
            if vi:
                return vi.strip()
            return en.replace("<en>", "").strip()
        return None

    def segment_oov(self, word: str, lang: str) -> str | None:
        cache_key = (word, lang)
        if cache_key in self._segmentation_cache:
            return self._segmentation_cache[cache_key]

        chars = list(word)
        dp: list[str | None] = [None] * (len(chars) + 1)
        dp[0] = ""
        for i in range(len(chars)):
            if dp[i] is None:
                continue
            for j in range(len(chars), i, -1):
                segment = "".join(chars[i:j])
                if not _has_vowel_and_consonant(segment):
                    continue
                phone = self.resolve_segment_phone(segment, lang)
                if phone is None:
                    continue
                new_phone = phone if not dp[i] else f"{dp[i]} {phone}"
                if dp[j] is None:
                    dp[j] = new_phone

        result = dp[len(chars)]
        if len(self._segmentation_cache) >= 5_000:
            self._segmentation_cache.clear()
        self._segmentation_cache[cache_key] = result
        return result

    def char_fallback(self, content: str, lang: str) -> str:
        parts: list[str] = []
        for char in content:
            lower = char.lower()
            merged = self.cached_lookup_merged(lower)
            if merged is not None:
                parts.append(merged.replace("<en>", "").strip())
                continue
            common = self.cached_lookup_common(lower)
            if common is not None:
                vi, en = common
                phone = en if lang == "en" and en else vi or en
                parts.append(phone.replace("<en>", "").strip())
                continue
            parts.append(lower)
        return "".join(parts)

    def phonemize(self, text: str) -> str:
        tokens: list[Token] = []

        for match in RE_TOKEN.finditer(text):
            en_tag, word, punct = match.group(1), match.group(2), match.group(3)
            if en_tag is not None:
                content = RE_TAG_STRIP.sub("", en_tag).strip()
                for submatch in RE_TAG_CONTENT.finditer(content):
                    subword, subpunct = submatch.group(1), submatch.group(2)
                    if subword is not None:
                        lower = subword.lower()
                        phone_val = None
                        merged = self.cached_lookup_merged(lower)
                        if merged is not None:
                            phone_val = merged.replace("<en>", "").strip()
                        else:
                            common = self.cached_lookup_common(lower)
                            if common is not None and common[1]:
                                phone_val = common[1].replace("<en>", "").strip()
                        tokens.append(Token("en", subword, phone_val, True))
                    elif subpunct is not None:
                        tokens.append(Token("punct", subpunct, subpunct, True))
            elif word is not None:
                lower = word.lower()
                merged = self.cached_lookup_merged(lower)
                if merged is not None:
                    lang = "en" if "<en>" in merged else "vi"
                    tokens.append(Token(lang, word, merged.replace("<en>", "").strip(), False))
                    continue
                common = self.cached_lookup_common(lower)
                if common is not None:
                    vi, en = common
                    phone = f"\x1f{vi.strip()}\x1f{en.replace('<en>', '').strip()}\x1f"
                    tokens.append(Token("common", word, phone, False))
                    continue
                has_vi_accent = any(char in VI_ACCENTS for char in lower)
                tokens.append(Token("vi" if has_vi_accent else "en", word, None, False))
            elif punct is not None:
                tokens.append(Token("punct", punct, punct, False))

        self.propagate_language(tokens)

        result: list[str] = []
        for token in tokens:
            if token.lang == "punct":
                result.append(token.content)
                continue

            if token.phone is not None:
                phone = self.resolve_common_token_phone(token)
            else:
                lower = token.content.lower()
                phone = self.segment_oov(lower, token.lang)
                if phone is None:
                    phone = self.char_fallback(token.content, token.lang)
            result.append(phone.strip())

        joined = " ".join(result)
        return (
            joined.replace(" .", ".")
            .replace(" ,", ",")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ;", ";")
            .replace(" :", ":")
        )

    @staticmethod
    def resolve_common_token_phone(token: Token) -> str:
        if token.phone is None:
            return ""
        if token.phone.startswith("\x1f") and token.phone.endswith("\x1f"):
            inner = token.phone[1:-1]
            sep = inner.find("\x1f")
            if sep == -1:
                sep = len(inner)
            if token.lang == "en":
                phone = inner[sep + 1 :] if sep + 1 <= len(inner) else ""
                if token.content.lower() == "a" and not token.is_explicit_en:
                    return "ɐ"
                return phone
            return inner[:sep]

        if token.lang == "en" and token.content.lower() == "a" and not token.is_explicit_en:
            return "ɐ"
        return token.phone

    @staticmethod
    def propagate_language(tokens: list[Token]) -> None:
        i = 0
        token_count = len(tokens)
        while i < token_count:
            if tokens[i].lang != "common":
                i += 1
                continue

            start = i
            while i < token_count and tokens[i].lang == "common":
                i += 1
            end = i - 1

            left_anchor = None
            left_dist = 999
            for left in range(start - 1, -1, -1):
                if _is_stop_punct(tokens[left]):
                    break
                if tokens[left].lang in {"vi", "en"}:
                    left_anchor = tokens[left].lang
                    left_dist = start - left
                    break

            right_anchor = None
            right_dist = 999
            for right in range(end + 1, token_count):
                if _is_stop_punct(tokens[right]):
                    break
                if tokens[right].lang in {"vi", "en"}:
                    right_anchor = tokens[right].lang
                    right_dist = right - end
                    break

            if left_anchor is not None and right_anchor is not None:
                final_lang = right_anchor if right_dist <= left_dist else left_anchor
            elif left_anchor is not None:
                final_lang = left_anchor
            elif right_anchor is not None:
                final_lang = right_anchor
            else:
                final_lang = "vi"

            for index in range(start, end + 1):
                tokens[index].lang = final_lang


def _is_stop_punct(token: Token) -> bool:
    return len(token.content) == 1 and token.content in STOP_PUNCT


class G2P:
    def __init__(self, resource_path: str | Path | None = None) -> None:
        self._engine = Engine(resource_path or _resource_path())

    def convert(self, text: str | list[str], punc_norm: bool = False, **kwargs) -> str | list[str]:
        if isinstance(text, list):
            return [self.convert(item, punc_norm=punc_norm, **kwargs) for item in text]
        return self._engine.phonemize(apply_punc_norm(text) if punc_norm else text)
