from __future__ import annotations

import unittest

from vig2p import VietnameseG2P, fix_phonemes, phonemize_many, phonemize_text, tokenize_text


class FakeBackend:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.calls: list[str] = []

    def run(self, text: str) -> str:
        self.calls.append(text)
        return self.mapping[text]


class Vig2PTest(unittest.TestCase):
    def test_fix_phonemes_maps_seag2p_symbols_to_kokoro_symbols(self):
        raw = "sˈin tʃˈaː2w, ɗˈɛ6p t̪ˈiɛɜŋ tˈe-ɲ mˈaː5 mˈaː6 ˈi ' ’ ‘ */& – ˈɛm ʐ"

        fixed = fix_phonemes(raw)

        self.assertIn("sˈin ʧˈaː↘w, dˈɛʔ↓p tˈiɛ↗ŋ tˈæɲ", fixed)
        self.assertIn("mˈaːʔ↗ mˈaːʔ↓", fixed)
        self.assertIn("—", fixed)
        self.assertTrue(fixed.endswith("ˈɛm ʒ"))
        for forbidden in ["tʃ", "ɗ", "t̪", "e-", "2", "5", "6", "'", "’", "‘", "*", "/", "&", "ʐ"]:
            self.assertNotIn(forbidden, fixed)

    def test_tokenize_text_preserves_words_spaces_and_punctuation(self):
        self.assertEqual(tokenize_text("Mình cần UI/UX – ổn."), ["Mình", " ", "cần", " ", "UI", "/", "UX", " ", "–", " ", "ổn", "."])

    def test_phonemize_text_uses_one_backend_for_words_only(self):
        backend = FakeBackend({"Xin": "sˈin", "chào": "tʃˈaː2w"})

        phonemes = phonemize_text("Xin chào!", backend=backend)

        self.assertEqual(phonemes, "sˈin ʧˈaː↘w!")
        self.assertEqual(backend.calls, ["Xin", "chào"])

    def test_phonemize_many_reuses_backend(self):
        backend = FakeBackend({"một": "mˈo6t", "hai": "hˈaːj"})

        self.assertEqual(phonemize_many(["một", "hai"], backend=backend), ["mˈoʔ↓t", "hˈaːj"])
        self.assertEqual(backend.calls, ["một", "hai"])

    def test_vietnamese_g2p_wrapper_uses_injected_backend(self):
        converter = VietnameseG2P(FakeBackend({"cách": "kˈe-3c"}))

        self.assertEqual(converter("cách"), "kˈæ↗c")

    def test_text_aware_t_and_th_contrast(self):
        backend = FakeBackend({"tường": "t̪ˈyə2ŋ", "thường": "tˈyə2ŋ", "teo": "t̪ˈɛw", "theo": "tˈɛw"})

        self.assertEqual(phonemize_text("tường", backend=backend), "tˈyə↘ŋ")
        self.assertEqual(phonemize_text("thường", backend=backend), "θˈyə↘ŋ")
        self.assertEqual(phonemize_text("teo", backend=backend), "tˈɛw")
        self.assertEqual(phonemize_text("theo", backend=backend), "θˈɛw")

    def test_text_aware_tr_and_ch_contrast(self):
        backend = FakeBackend({"trước": "tʃˈyə3c", "chước": "tʃˈyə3c"})

        self.assertEqual(phonemize_text("trước", backend=backend), "ʈʂˈyə↗c")
        self.assertEqual(phonemize_text("chước", backend=backend), "ʧˈyə↗c")

    def test_text_aware_s_and_x_contrast(self):
        backend = FakeBackend({"số": "sˈo3", "xố": "sˈo3", "sinh": "sˈiɲ", "xinh": "sˈiɲ"})

        self.assertEqual(phonemize_text("số", backend=backend), "ʂˈo↗")
        self.assertEqual(phonemize_text("xố", backend=backend), "sˈo↗")
        self.assertEqual(phonemize_text("sinh", backend=backend), "ʂˈiɲ")
        self.assertEqual(phonemize_text("xinh", backend=backend), "sˈiɲ")

    def test_text_aware_gi_and_d_contrast(self):
        backend = FakeBackend({"giải": "zˈaː4j", "dải": "zˈaː4j", "gì": "zˈi2", "dì": "zˈi2"})

        self.assertEqual(phonemize_text("giải", backend=backend), "ʝˈaː↓j")
        self.assertEqual(phonemize_text("dải", backend=backend), "zˈaː↓j")
        self.assertEqual(phonemize_text("gì", backend=backend), "ʝˈi↘")
        self.assertEqual(phonemize_text("dì", backend=backend), "zˈi↘")

    def test_english_words_are_not_rewritten_as_vietnamese_contrasts(self):
        backend = FakeBackend({"team": "tˈiːm", "start": "stˈɑːɹt", "style": "stˈaɪl", "travel": "tɹˈævəl", "giant": "dʒˈaɪənt"})

        phonemes = phonemize_text("team start style travel giant", backend=backend)

        self.assertEqual(phonemes, "tˈiːm stˈɑːɹt stˈaɪl tɹˈævəl dʒˈaɪənt")
        self.assertNotIn("θˈiːm", phonemes)
        self.assertNotIn("ʈʂ", phonemes)
        self.assertNotIn("ʝ", phonemes)

    def test_mixed_text_punctuation_is_normalized(self):
        backend = FakeBackend({
            "Mình": "mˈi2ɲ",
            "cần": "kˈən2",
            "budget": "bˈʌdʒᵻt",
            "UI": "jˌuːˈaɪ",
            "UX": "jˌuːˈɛks",
            "don't": "dˈoʊnt",
            "panic": "pˈænɪk",
            "thường": "tˈyə2ŋ",
        })

        phonemes = phonemize_text("Mình cần **budget** UI/UX & don’t panic – thường.", backend=backend)

        self.assertNotIn("*", phonemes)
        self.assertNotIn("/", phonemes)
        self.assertNotIn("&", phonemes)
        self.assertIn("—", phonemes)
        self.assertIn("θˈyə↘ŋ", phonemes)


if __name__ == "__main__":
    unittest.main()
